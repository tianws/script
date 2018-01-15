#!/usr/bin/env python
# coding=utf-8
from __future__ import division, unicode_literals

import sys
sys.path.append('..')

from common import caffe_directory, find_sample, video_generator
from common import Path
from common import Rect
import cv2
import logging
from phodopus import is_same_target
import random
import subprocess

pos_label = 0
neg_label = 1
min_width = 200
min_height = 200
overlap_rate = 0


def _find_neg_rect(image, pos_rects):
    image_height, image_width, _ = image.shape
    if image_width < min_width or image_height < min_height:
        return None
    loop_times = 15
    for loop_time in range(loop_times):
        width = random.randint(min_width, image_width)
        x = random.randint(0, image_width - width)
        height = random.randint(min_height, image_height)
        y = random.randint(0, image_height - height)
        random_rect = Rect(x, y, width, height)
        duplicate_rects = [
                rect
                for rect in pos_rects
                if is_same_target(random_rect, rect, overlap_rate)]
        if len(duplicate_rects) == 0:
            return random_rect
    return None


def _handle_video(
        label_pathname,
        sample_pathname,
        caffe_dataset_directory,
        caffe_label_file):
    label_dict = {}
    for line in label_pathname.open():
        frame_filename_str, rects_str = line.rstrip('\n').split(' ', 1)
        frame_filename_str = frame_filename_str.split('.')[0]
        rects = [Rect(rect_str) for rect_str in rects_str.split(' ')]
        label_dict[frame_filename_str] = rects
    video = video_generator(sample_pathname)
    for frame_filename_str, frame in video:
        frame_filename_str = frame_filename_str.split('.')[0]
        if frame_filename_str in label_dict:
            rects = label_dict[frame_filename_str]
            for rect_index, rect in enumerate(rects):
                image_filename = '{}-{}-{}.jpg'.format(
                        sample_pathname.stem,
                        frame_filename_str,
                        rect_index)
                x, y, width, height = tuple(rect)
                image = frame[y:y + height, x:x + width]
                cv2.imwrite(
                        '{}/{}'.format(
                                caffe_dataset_directory, image_filename),
                        image)
                caffe_label_file.write(
                        '{} {}\n'.format(image_filename, pos_label))
                logging.debug('handle image: {}'.format(image_filename))
            neg_rect = _find_neg_rect(frame, rects)
            if neg_rect is not None:
                x, y, width, height = tuple(neg_rect)
                neg_image = frame[y:y + height, x:x + width]
                neg_image_filename = '{}-{}-{}.jpg'.format(
                        sample_pathname.stem,
                        frame_filename_str,
                        len(rects))
                cv2.imwrite(
                        '{}/{}'.format(
                                caffe_dataset_directory, neg_image_filename),
                        neg_image)
                caffe_label_file.write(
                        '{} {}\n'.format(neg_image_filename, neg_label))
                logging.debug(
                        'handle neg_image: {}'.format(neg_image_filename))


def handle_labels(
        dataset_directory, caffe_dataset_directory, caffe_label_pathname):
    with caffe_label_pathname.open('w') as caffe_label_file:
        for label_pathname in [
                pathname
                for pathname in dataset_directory.iterdir()
                if pathname.suffix == '.txt']:
            sample_pathname = find_sample(label_pathname)
            _handle_video(
                    label_pathname,
                    sample_pathname,
                    caffe_dataset_directory,
                    caffe_label_file)
            logging.info('handle {}\'s label done'.format(sample_pathname))


def handle_lmdb(caffe_dataset_directory, label_pathname, lmdb_pathname):
    command_pathname = caffe_directory / 'build/tools/convert_imageset'
    encode_type = 'jpg'
    command = [
            command_pathname,
            '-encode_type', encode_type,
            '-resize_height', min_height,
            '-resize_width', min_width,
            '-shuffle',
            '{}/'.format(caffe_dataset_directory),
            # this pathname must have a extra slash because of
            # convert_imageset's bad design
            label_pathname,
            lmdb_pathname,
    ]
    command = list(map(str, command))
    logging.info(' '.join(command))
    subprocess.call(command)
    logging.info('handle lmdb done: {}'.format(caffe_dataset_directory))


def main():
    directory = Path('/home/acgtyrant/BigDatas/car')
    trainset_directory = directory / 'trainset'
    caffe_trainset_directory = directory / 'caffe_trainset'
    if not caffe_trainset_directory.exists():
        caffe_trainset_directory.mkdir()
    train_label_pathname = directory / 'train.txt'
    train_lmdb_pathname = directory / 'train_lmdb'
    handle_labels(
            trainset_directory,
            caffe_trainset_directory,
            train_label_pathname)
    handle_lmdb(
            caffe_trainset_directory,
            train_label_pathname,
            train_lmdb_pathname)
    testset_directory = directory / 'testset'
    caffe_testset_directory = directory / 'caffe_testset'
    if not caffe_testset_directory.exists():
        caffe_testset_directory.mkdir()
    test_label_pathname = directory / 'test.txt'
    test_lmdb_pathname = directory / 'test_lmdb'
    handle_labels(
            testset_directory,
            caffe_testset_directory,
            test_label_pathname)
    handle_lmdb(
            caffe_testset_directory,
            test_label_pathname,
            test_lmdb_pathname)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    main()
