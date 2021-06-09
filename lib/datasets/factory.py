# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from lib.datasets.pascal_voc import pascal_voc
from lib.datasets.coco import coco
from lib.datasets.DIY_pascal_voc import DIY_pascal_voc
from lib.datasets.casia_mask import casia
from lib.datasets.coverage_mask import coverage
from lib.datasets.columbia_mask import columbia
# from lib.datasets.Nist16 import Nist16
from lib.datasets.Nist16_mask import Nist16_3  #3 type,dist_NIST_train_new_6
from lib.datasets.Nist16_mask_1 import Nist16 #one type
import numpy as np

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

for year in ['2012']:
    # for split in ['trainval']:
    for split in ['test']:
        name = 'DIY_dataset'
        __sets[name] = (lambda split=split, year=year: DIY_pascal_voc(split, year))
# print(__sets['voc_2012_trainval']())
# coco_path='E:\data\图像篡改\coco_synthetic'
coco_path='/data/cxm/data/coco_synthetic'
for split in ['coco_train_filter_single', 'coco_test_filter_single']:
    name = split
    __sets[name] = (lambda split=split: coco(split,2007,coco_path))
# print(__sets['coco_train_filter_single']())

# casia_path='E:\data\CASIA2\Tp'
# casia_path='E:\data\casia\CASIA1\Sp'
# casia_path='/data/cxm/data/CASIA2/Tp'
# casia_path='/data/cxm/data/CASIA2/Tp_new'
# casia_path='/data/cxm/data/CASIA1/Tp_new'
casia_path='E:\Server_backup\data\CASIA'
# casia_path='/data/cxm/data/CASIA 1.0 dataset/Tp/TP'
# casia_path='E:\data\COVERAGE\image'
#for split in ['casia_train_all_single', 'casia_test_all_1']:
for split in ['casia_train_all_single', 'casia_test_all_single']:
    name = split
    __sets[name] = (lambda split=split: casia(split,2007,casia_path))

coverage_path=r'E:\Server_backup\data\coverage\\'
for split in ['coverage_train_single', 'coverage_test_single']:
    name = split
    __sets[name] = (lambda split=split: coverage(split,2007,coverage_path))

# columbia_path='/data/cxm/data/4cam_splc'
columbia_path='E:\Server_backup\data\Columbia'
for split in ['columbia_train_all_single', 'columbia_test_all_single']:
    name = split
    __sets[name] = (lambda split=split: columbia(split,2007,columbia_path))

nist_path=r'E:\Server_backup\data\NC2016_Test0601'
# for split in ['dist_NIST_train_new_6', 'dist_NIST_test_new_6']:#3type
for split in ['Nist16_train_all_single', 'Nist16_test_all_single']:#one type
    name = split
    __sets[name] = (lambda split=split: Nist16_3(split,2007,nist_path))

nist1_path=r'E:\\Server_backup\\data\\NC2016_Test0601'
# for split in ['dist_NIST_train_new_6', 'dist_NIST_test_new_6']:#3type
for split in ['Nist16_train_all_single', 'Nist16_test_all_single']:#one type
    name = split
    __sets[name] = (lambda split=split: Nist16(split,2007,nist1_path))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
