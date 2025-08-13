# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate ADE20K description file'
    )
    parser.add_argument(
        'data_root',
        type=Path,
        help='Path to the data root directory, i.e. ADE20K_2021_17_01',
    )
    parser.add_argument(
        'output_file',
        type=Path,
        help='Path to the output file',
    )
    parser.add_argument(
        '--split',
        choices=['training', 'validation', 'test'],
        default='training',
        help='Specify a split of the dataset to process (default: training)',
    )
    parser.add_argument('--version', choices=['2016', '2021'], default='2021')
    return parser.parse_args()


def _job2021(img):
    objects = {}
    gt_seg = img.with_name(f"{img.stem}_seg.png")
    desc_file = img.with_suffix('.json')
    if not desc_file.exists():
        return
    with desc_file.open(encoding='cp1252') as f:
        desc = json.load(f)
    seg = np.array(Image.open(gt_seg))
    for obj in desc['annotation']['object']:
        high_val = (seg[..., 0] // 10).astype('int32')
        low_val = seg[..., 1].astype('int32')
        seg_idx = high_val * 256 + low_val
        palettes = seg[seg_idx == obj['name_ndx']]
        if palettes.size == 0:
            continue
        palette = palettes[0]
        objects[obj['name']] = dict(
            id=obj['name_ndx'],
            palette=palette.tolist()
        )
    return objects


def generate_2021(args):
    objects = {}
    pool = ProcessPoolExecutor()
    events = []
    for img in Path(args.data_root).rglob(f"**/{args.split}/**/*.jpg"):
        event = pool.submit(_job2021, img)
        events.append(event)
    for i, event in enumerate(events):
        result = event.result()
        objects.update(result)
        if i % 100 == 0:
            print(f"Processed {i:05d} images", end='\r')
    print('\nDone')
    objects_sorted = dict(sorted(objects.items(), key=lambda x: x[1]['id']))
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(objects_sorted, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    if args.version == '2021':
        generate_2021(args)
    else:
        raise NotImplementedError(
            'ADE20K dataset converter for version 2016 is not implemented'
        )
