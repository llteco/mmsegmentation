# Copyright (c) OpenMMLab. All rights reserved.
import json
from pathlib import Path
from typing import List, Literal

import mmengine.fileio as fileio
import pooch

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ADE20KDataset(BaseSegDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=('wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road',
                 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk',
                 'person', 'earth', 'door', 'table', 'mountain', 'plant',
                 'curtain', 'chair', 'car', 'water', 'painting', 'sofa',
                 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair',
                 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
                 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
                 'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
                 'skyscraper', 'fireplace', 'refrigerator', 'grandstand',
                 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow',
                 'screen door', 'stairway', 'river', 'bridge', 'bookcase',
                 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill',
                 'bench', 'countertop', 'stove', 'palm', 'kitchen island',
                 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine',
                 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
                 'chandelier', 'awning', 'streetlight', 'booth',
                 'television receiver', 'airplane', 'dirt track', 'apparel',
                 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle',
                 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain',
                 'conveyer belt', 'canopy', 'washer', 'plaything',
                 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall',
                 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food',
                 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal',
                 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket',
                 'sculpture', 'hood', 'sconce', 'vase', 'traffic light',
                 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate',
                 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
                 'clock', 'flag'),
        palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                 [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                 [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                 [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                 [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                 [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                 [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                 [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                 [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                 [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                 [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                 [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                 [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                 [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                 [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                 [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                 [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                 [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                 [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                 [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                 [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                 [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                 [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                 [102, 255, 0], [92, 0, 255]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class ADE20K2016Dataset(ADE20KDataset):
    r"""Alias to ADE20KDataset."""


@DATASETS.register_module()
class ADE20K2021Dataset(BaseSegDataset):
    """ADE20K dataset for 2021_17_01 version without pre-processing.

    .. code-block:: python

        R = seg[:,:,0]
        G = seg[:,:,1]
        B = seg[:,:,2]
        ObjectClassMasks = (R/10).astype(np.int32)*256+(G.astype(np.int32))
    """

    ANN_FILE = (
        r'https://github.com/llteco/mmsegmentation/releases/download/'
        r'v1.2.5/ade20k_2021_17_01.json'
    )  # ann file generated by tools/dataset_converters/ade20k.py
    ANN_HASH = (
        'sha256:'
        '7fd0eaf00e039d36c7e18e0b20fde601606ce4472bb46f7e7e1a105621a88b5b'
    )
    METAINFO = {}  # to fill dynamically

    @classmethod
    def default_metainfo(cls, ann_file=None, known_hash=None) -> dict:
        """Fill default metainfo from ann_file."""
        ann_file = ann_file or cls.ANN_FILE
        if fileio.get_file_backend(ann_file).name == 'HTTPBackend':
            ann_file = pooch.retrieve(
                    url=ann_file,
                    known_hash=known_hash or cls.ANN_HASH,
                )
        desc = json.loads(fileio.get(ann_file))
        metainfo = {}
        metainfo['classes'] = tuple(
            k.split(',', 1)[0] for k in desc.keys()
        )
        metainfo['palette'] = tuple(
            v['palette'] for v in desc.values()
        )
        return metainfo

    def __init__(
        self,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        reduce_zero_label=True,
        split: Literal['training', 'validation', 'test'] = 'training',
        **kwargs,
    ):
        self.split = split
        ann_file = kwargs.pop('ann_file', self.ANN_FILE)
        metainfo = self.default_metainfo(ann_file, kwargs.get('ann_hash'))
        assert reduce_zero_label
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ann_file=ann_file,
            metainfo=metainfo,
            **kwargs,
        )

    def load_data_list(self) -> list[dict]:
        """ADE20K 2021_17_01 data structure:

        |-images/ADE/training
        |  |-<super-cat> (home_or_hotel)
        |  |  |-<sub-cat> (alcove)
        |  |  |  |-ADE_train_{id:08d}.jpg
        |  |  |  |-ADE_train_{id:08d}_seg.png
        |  |  |  |-ADE_train_{id:08d}.json
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', f"images/ADE/{self.split}")
        label_map = {}
        for i, color in enumerate(self.metainfo['palette']):
            r, g, _ = color
            label_map[r // 10 * 256 + g] = i
        for img in fileio.list_dir_or_file(
            dir_path=img_dir,
            list_dir=False,
            suffix=self.img_suffix,
            recursive=True,
            backend_args=self.backend_args,
        ):
            img_url = Path(img_dir) / img
            data_info: dict = dict(img_path=img_url)
            seg_map = img_url.with_name(
                f"{img_url.stem}_seg{self.seg_map_suffix}")
            if seg_map.exists():
                data_info['seg_map_path'] = seg_map
            desc_file = img_url.with_suffix('.json')
            try:
                with open(desc_file, encoding='utf-8') as f:
                    desc = json.load(f)['annotation']
            except UnicodeDecodeError:
                with open(desc_file, encoding='cp1252') as f:
                    desc = json.load(f)['annotation']
            data_info['label_map'] = label_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_info['_scene'] = desc['scene']
            data_info['_objects'] = {i['name_ndx'] for i in desc['object']}
            data_list.append(data_info)
        data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list

    def filter_data(self) -> List[dict]:
        return self.data_list
