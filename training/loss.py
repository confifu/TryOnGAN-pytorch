# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

    def run_G(self, z, c, pose, ret_pose = True, sync=True):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            pose, bin_regions, col_regions, img = self.G_synthesis(ws, pose, ret_pose= ret_pose)
        return pose, bin_regions, col_regions, img, ws

    def run_D(self, img, pose, bin_regions, col_regions, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, pose, bin_regions, col_regions, c)
        return logits
    
    def getBinRegionDict(self, pmap, detach = False, require_grad = False):
        '''
        returns 32x32 bin map for 6 region
        '''
        #reshape pmap to 32x32
        pmap = torch.nn.functional.interpolate(pmap,
                                    size=(64, 64),
                                    mode='bilinear',
                                    align_corners=True)
        hi = torch.normal(1, 0.1, size=pmap.shape).cuda()
        lo = torch.normal(-1, 0.1, size=pmap.shape).cuda()
        hi = hi.requires_grad_(require_grad)
        lo = lo.requires_grad_(require_grad)
        if detach:
            pmap = pmap.detach()
            pmap = pmap.requires_grad_(require_grad)
        region = {}
        region[0] = torch.where(0 == pmap, hi, lo)
        region[1] = torch.where(1 == pmap, hi, lo)
        region[2] = torch.where(2 == pmap, hi, lo)
        region[3] = torch.where(3 == pmap, hi, lo)
        region[4] = torch.where(4 == pmap, hi, lo)
        region[5] = torch.where(5 == pmap, hi, lo)
        region[6] = torch.where(6 == pmap, hi, lo)
        return region

    def getColRegionDict(self, img, pmap, detach = False, require_grad= False):
        '''
        res = img.shape[-1]
        pmap = torch.nn.functional.interpolate(pmap,
                                    size=(res//2, res//2),
                                    mode='bilinear',
                                    align_corners=True)
        img = torch.nn.functional.interpolate(img,
                                    size=(res//2, res//2),
                                    mode='bilinear',
                                    align_corners=True)
        '''
        pmap = torch.cat([pmap, pmap, pmap], dim = 1)
        lo = torch.normal(-1, 0.1, size=img.shape).cuda()
        lo = lo.requires_grad_(require_grad)
        if detach:
            pmap = pmap.detach().requires_grad_(require_grad)
            img = img.detach().requires_grad_(require_grad)
        region = {}
        region[0] = torch.where(0 == pmap, img, lo)
        region[1] = torch.where(1 == pmap, img, lo)
        region[2] = torch.where(2 == pmap, img, lo)
        region[3] = torch.where(3 == pmap, img, lo)
        region[4] = torch.where(4 == pmap, img, lo)
        region[5] = torch.where(5 == pmap, img, lo)
        region[6] = torch.where(6 == pmap, img, lo)
        return region

    def accumulate_gradients(self, phase, real_img, real_pmap, real_pose, real_c, gen_z, gen_c, sync, gain, traintill):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        assert traintill in ['all', 'bin', 'col']
        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_pose, gen_bin_regions, gen_col_regions, gen_img, _gen_ws = self.run_G(gen_z, gen_c, real_pose, ret_pose = True, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                real_bin_regions = self.getBinRegionDict(real_pmap, detach = False, require_grad=False)
                real_col_regions = self.getColRegionDict(real_img, real_pmap, detach = False, require_grad=False)

                if traintill is 'all':
                    gen_logits = self.run_D(gen_img, gen_pose, gen_bin_regions, gen_col_regions, gen_c, sync=False)
                elif traintill is 'col':
                    gen_logits = self.run_D(real_img, gen_pose, gen_bin_regions, gen_col_regions, gen_c, sync=False)
                elif traintill is 'bin':
                    gen_logits = self.run_D(real_img, gen_pose, gen_bin_regions, real_col_regions, gen_c, sync=False)

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
            #with torch.autograd.profiler.record_function('Gmain_backward'):
            with torch.autograd.set_detect_anomaly(True):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_pose, gen_bin_regions, gen_col_regions, gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], real_pose[:batch_size], ret_pose = True, sync=sync)
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    outputs = []

                    if traintill is 'all':
                        outputs = [(gen_img * torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])).sum()]
                    if traintill is 'col' or traintill is 'all':
                        outputs.extend([(gen_col_regions[i] * torch.randn_like(gen_col_regions[i]) / np.sqrt(gen_col_regions[i].shape[2] * gen_col_regions[i].shape[3])).sum() for i in range(7)])
                    if traintill is 'col' or traintill is 'all' or traintill is 'bin':
                        outputs.extend([(gen_bin_regions[i] * torch.randn_like(gen_bin_regions[i]) / np.sqrt(gen_bin_regions[i].shape[2] * gen_bin_regions[i].shape[3])).sum() for i in range(7)])

                    #outputs.extend([(gen_pose * torch.randn_like(gen_pose) / np.sqrt(gen_pose.shape[2] * gen_pose.shape[3])).sum()])
                    pl_grads = torch.autograd.grad(outputs=outputs, inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_pose, gen_bin_regions, gen_col_regions, gen_img, _gen_ws = self.run_G(gen_z, gen_c, real_pose, ret_pose = True, sync=False)
                real_bin_regions = self.getBinRegionDict(real_pmap, detach = False, require_grad=False)
                real_col_regions = self.getColRegionDict(real_img, real_pmap, detach = False, require_grad=False)

                if traintill is 'all':
                    gen_logits = self.run_D(gen_img, gen_pose, gen_bin_regions, gen_col_regions, gen_c, sync=False) # Gets synced by loss_Dreal.
                elif traintill is 'col':
                    gen_logits = self.run_D(real_img, gen_pose, gen_bin_regions, gen_col_regions, gen_c, sync=False)
                elif traintill is 'bin':
                    gen_logits = self.run_D(real_img, gen_pose, gen_bin_regions, real_col_regions, gen_c, sync=False)

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_pose_tmp = real_pose.detach().requires_grad_(do_Dr1)
                real_bin_region_tmp = self.getBinRegionDict(real_pmap, detach = True, require_grad=do_Dr1)
                real_col_region_tmp = self.getColRegionDict(real_img_tmp, real_pmap, detach = True, require_grad=do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_pose_tmp, real_bin_region_tmp, real_col_region_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        try:
                            inputs = [real_img_tmp, real_pose_tmp]
                            inputs.extend([real_bin_region_tmp[i] for i in range(7)])
                            inputs.extend([real_col_region_tmp[i] for i in range(7)])
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=inputs, create_graph=True, only_inputs=True)[0]
                        except:
                            import traceback
                            traceback.print_exc()
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
