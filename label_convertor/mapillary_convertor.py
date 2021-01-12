#!/usr/bin/env python
# encoding: utf-8
"""
@author: tianws
@file: mapillary_convertor.py
@time: 2021/1/12 下午2:12
"""
# coding=utf-8

import numpy as np
from PIL import Image
from pathlib import Path
import json
from pprint import pprint
import argparse

# original_label_id : new_label_id
CONVERTOR_DICT = {24: 1, 49: 2, 50: 2, 45: 3, 46: 3, 47: 3}

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image_dir', help='input image dir',
                    default='/DATA2/Dataset/mapillary-vistas-dataset_public_v1.1/validation/images')
parser.add_argument('-l', '--label_dir', help='input label dir',
                    default='/DATA2/Dataset/mapillary-vistas-dataset_public_v1.1/validation/labels')
parser.add_argument('-o', '--out_dir', help='output image dir', default='example')
parser.add_argument('-r', '--redering', type=int, help='render or not，rendering if 1, default is 1', default=1)
parser.add_argument('-b', '--background_id', type=int, help='the id of background, default is 0', default=0)

args = parser.parse_args()

# parse args
image_root = Path(args.image_dir)
label_root = Path(args.label_dir)
result_root = Path(args.out_dir)

redering = args.redering
background_id = args.background_id


def create_foloder(result_root, flag):
    dirs = result_root / flag
    dirs.mkdir(parents=True, exist_ok=True)
    return dirs


def get_color_map(config_json_path='config.json'):
    with open(config_json_path) as config_file:
        config = json.load(config_file)
    config_labels = config['labels']
    # calculate label color mapping
    color_map = []
    trainid_to_name = {}
    for i in range(0, len(config_labels)):
        if i == background_id:
            color = [0, 0, 0]
        else:
            color = config_labels[i]['color']
        color_map = color_map + color
        name = config_labels[i]['readable']
        name = name.replace(' ', '_')
        trainid_to_name[i] = name
    return color_map, trainid_to_name


def convert_label(original_label_arr, convertor_dict):
    new_label_arr = np.zeros(original_label_arr.shape, dtype=np.uint8)
    if background_id != 0:
        new_label_arr[:] = background_id
    for picked_original_id, picked_new_id in convertor_dict.items():
        new_label_arr[original_label_arr == picked_original_id] = picked_new_id
    return new_label_arr


def render(image_arr, new_label_arr, color_map, alpha=0.7):
    color_label_pil = Image.fromarray(new_label_arr)
    color_label_pil.putpalette(color_map)
    color_label_pil_RGB = color_label_pil.convert('RGB')
    color_label_arr = np.array(color_label_pil_RGB)

    mixed_mask = color_label_arr * alpha + image_arr * (1 - alpha)
    mixed_mask = mixed_mask.astype(np.uint8)

    mixed_arr = np.where(new_label_arr[:, :, np.newaxis] == background_id, image_arr, mixed_mask)
    mixed_pil = Image.fromarray(mixed_arr)

    return color_label_pil, mixed_pil


def main():
    color_map, trainid_to_name = get_color_map()
    print('original id to name:')
    pprint(trainid_to_name)

    # create result folder
    grey_result_dir = create_foloder(result_root, 'grey')
    print('Save gray result to: ', grey_result_dir)
    if redering:
        color_result_dir = create_foloder(result_root, 'color')
        mixed_result_dir = create_foloder(result_root, 'mixed')
        print('Save color result to: ', color_result_dir)
        print('Save mixed result to: ', mixed_result_dir)

    count = 0

    for label_path in label_root.iterdir():
        if label_path.suffix not in ['.jpg', '.png']:
            continue

        count = count + 1
        print('Processing', count, label_path)

        # Path
        # image name: *.jpg
        image_name = label_path.stem
        image_path = image_root.joinpath(image_name + '.jpg')
        # label name: *.png
        label_name = label_path.name
        grey_result_path = grey_result_dir / label_name
        if redering:
            color_result_path = color_result_dir / label_name
            mixed_result_path = mixed_result_dir / label_name

        # Read image and label
        image_pil = Image.open(str(image_path))
        image_arr = np.array(image_pil)

        original_label_pil = Image.open(str(label_path))
        original_label_arr = np.array(original_label_pil)

        new_label_arr = convert_label(original_label_arr, CONVERTOR_DICT)
        new_label_pil = Image.fromarray(new_label_arr)
        new_label_pil.save(str(grey_result_path))
        if redering:
            color_label_pil, mixed_pil = render(image_arr, new_label_arr, color_map)
            color_label_pil.save(str(color_result_path))
            mixed_pil.save(str(mixed_result_path))
        image_pil.close()
        original_label_pil.close()


if __name__ == '__main__':
    main()
