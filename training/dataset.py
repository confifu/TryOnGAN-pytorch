﻿# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import pandas as pd
from scipy.stats import multivariate_normal

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_parsemap(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def get_pose(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8

        parsemap = self._load_parsemap(self._raw_idx[idx])
        assert isinstance(parsemap, np.ndarray)
        #assert list(parsemap.shape) == self.image_shape
        assert parsemap.dtype == np.uint8

        pose = self.get_pose()

        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]

            assert parsemap.ndim == 3 # CHW
            parsemap = parsemap[:, :, ::-1]

            pose = torch.flip(pose, [-1])
        
        return image.copy(), parsemap.copy(), self.get_label(idx), pose

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        parsepath,              # Path to zip for parse maps
        pose_file,              # Path to csv file with pose keypoints
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._parsepath = parsepath
        self._zipfile = None
        self._parse_zipfile = None

        #load RGB image fnames
        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile(self._path).namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        #load parse file names
        if os.path.isdir(self._parsepath):
            self._type = 'dir'
            self._all_parse_fnames = {os.path.relpath(os.path.join(root, fname), start=self._parsepath) for root, _dirs, files in os.walk(self._parsepath) for fname in files}
        elif self._file_ext(self._parsepath) == '.zip':
            self._type = 'zip'
            self._all_parse_fnames = set(self._get_zipfile(self._parsepath).namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        self._parse_fnames = sorted(fname for fname in self._all_parse_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._parse_fnames) == 0:
            raise IOError('No image files found in the specified path')
        

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')

        self.df = pd.read_csv(pose_file)
        self.image_size = resolution
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self, path):
        assert self._type == 'zip'
        if path == self._path:
            if self._zipfile is None:
                self._zipfile = zipfile.ZipFile(path)
            return self._zipfile
        elif path == self._parsepath:
            if self._parse_zipfile is None:
                self._parse_zipfile = zipfile.ZipFile(path)
            return self._parse_zipfile
        

    def _open_file(self, fname, path):
        if self._type == 'dir':
            return open(os.path.join(path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile(path).open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

        try:
            if self._parse_zipfile is not None:
                self._parse_zipfile.close()
        finally:
            self._parse_zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None, _parse_zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        self.fname = fname
        with self._open_file(fname, self._path) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_parsemap(self, raw_idx):
        fname = self._parse_fnames[raw_idx]
        with self._open_file(fname, self._parsepath) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname, self._path) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def get_pose(self):
        base = os.path.basename(self.fname)
        keypoint_list = self.df[self.df['name'] == base]['keypoints'].tolist()
        if len(keypoint_list) > 0:
            keypoint = keypoint_list[0]
            ptlist = keypoint.split(':')
            ptlist = [float(x) for x in ptlist]
            heatmap = self.getHeatMap(ptlist)
            return heatmap
        else:
            heatmap = torch.zeros(17, 32, 32)
            return heatmap

    def getHeatMap(self, pose):
        '''
        pose should be a list of length 51, every 3 number for
        x, y and confidence for each of the 17 keypoints.
        '''

        stack = []
        for i in range(17):
            x = pose[3*i]
            
            y = pose[3*i + 1]
            c = pose[3*i + 2]
            
            ratio = 16.0 / self.image_size
            map = self.getGaussianHeatMap([x*ratio, y*ratio])

            if c < 0.4:
                map = 0.0 * map
            stack.append(map)
        
        maps = np.dstack(stack)
        heatmap = torch.from_numpy(maps).transpose(0, -1)
        return heatmap

    def getGaussianHeatMap(self, bonePos):
        width = 32
        x, y = np.mgrid[0:width:1, 0:width:1]
        pos = np.dstack((x, y))

        gau = multivariate_normal(mean = list(bonePos), cov = [[width*0.02, 0.0], [0.0, width*0.02]]).pdf(pos)
        return gau

#----------------------------------------------------------------------------
