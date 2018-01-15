#!/usr/bin/env python
# coding=utf-8
from __future__ import division, unicode_literals

from common import Path, Rect
from common import video_generator
import cv2
import math
import logging
import os
import phodopus
import shutil
import subprocess
import time

directory = Path(os.path.expanduser('~')) / 'BigDatas/car'
cascade_directory = directory / 'cascade'
cascade_pathname = cascade_directory / 'cascade.xml'
params_pathname = cascade_directory / 'params.xml'
info_pathname = directory / 'scaled_positive_samples.txt'
num_stages = (15)
width, height = 20, 20
positive_samples_count = sum(map(int, [
        line.split(' ', 2)[1] for line in info_pathname.open()]))
feature_type = 'LBP'
max_weak_count = 1000
vec_pathname = directory / 'positive_samples.vec'
background_directory = directory / 'background'
samples_directory = directory / 'samples'
trainset_directory = directory / 'trainset'


def detect(video_pathname, cascade_pathname, scale_down_ratio, log_file):
    video = video_generator(video_pathname)
    cascade = cv2.CascadeClassifier(str(cascade_pathname))
    for frame_filename_str, frame in video:
        frame = cv2.resize(frame, (20, 20))
        rects = cascade.detectMultiScale(frame, 1.2, 3, 0, (20, 20))
        new_rects = []
        for x, y, width, height in rects:
            new_rect = Rect(
                    scale_down_ratio * x,
                    scale_down_ratio * y,
                    scale_down_ratio * width,
                    scale_down_ratio * height)
            new_rects.append(new_rect)
        rects_str = ' '.join([
                ','.join(map(str, rect))
                for rect in new_rects])
        log_file.write('{} {}\n'.format(frame_filename_str, rects_str))


def generate_vec():
    if not vec_pathname.exists():
        opencv_createsamples_command = [
                'opencv_createsamples',
                '-info', info_pathname,
                '-vec', vec_pathname,
                '-num', positive_samples_count,
                '-w', width,
                '-h', height,
        ]
        opencv_createsamples_command = list(
                map(str, opencv_createsamples_command))
        logging.info(' '.join(opencv_createsamples_command))
        subprocess.call(opencv_createsamples_command)


def generate_background(cascade_pathname, scale_down_ratio):
    background_pathname = directory / 'background.txt'
    for image_pathname in background_directory.iterdir():
        image_pathname.unlink()
    with background_pathname.open('w') as background_file:
        for pathname in trainset_directory.iterdir():
            if pathname.suffix == '' and Path(pathname / '0.jpg').exists():
                logging.debug('detect {}'.format(pathname))
                log_pathname = pathname.with_suffix('.log')
                label_pathname = pathname.with_suffix('.txt')
                fp_pathname = pathname.with_suffix('.fp')
                with log_pathname.open('w') as log_file:
                    detect(
                            pathname,
                            cascade_pathname,
                            scale_down_ratio,
                            log_file)
                phodopus.parse(
                        log_pathname,
                        label_pathname,
                        fp_pathname=fp_pathname)
                with fp_pathname.open() as fp_file:
                    for line in fp_file:
                        image_filename_str, rects_str = line.split(
                                ' ', 1)
                        for index, rect_str in enumerate(rects_str.split()):
                            rect = Rect(rect_str)
                            image = cv2.imread(
                                    str(pathname / image_filename_str))
                            image = image[
                                    rect.y:rect.y + rect.height,
                                    rect.x:rect.x + rect.width]
                            width = int(image.shape[1] / scale_down_ratio)
                            height = int(image.shape[0] / scale_down_ratio)
                            image = cv2.resize(image, (width, height))
                            new_image_filename_str = '{}-{}-{}.jpg'.format(
                                    pathname.name,
                                    image_filename_str.split('.')[0],
                                    index)
                            new_image_pathname = (
                                    background_directory /
                                    new_image_filename_str)
                            cv2.imwrite(
                                    str(new_image_pathname), image)
                            background_file.write('background/{}\n'.format(
                                    new_image_filename_str))
    shuf_command = [
            'shuf',
            str(background_pathname),
            '-o',
            str(background_pathname)]
    shuf_command = map(str, shuf_command)
    subprocess.call(shuf_command)


def main():
    generate_vec()
    min_hit_rate = 0.99
    max_false_alarm_rate = 0.5
    for num_stage in range(5, num_stages + 1):
        if num_stage == 5:
            background_pathname = directory / 'positive_samples.txt'
            num_positive = int(math.ceil(positive_samples_count / (
                    (1 - min_hit_rate) * num_stages + 2)))
            num_negative = num_positive * 2
        else:
            background_pathname = directory / 'background.txt'
            generate_background(
                    cascade_pathname, (num_stages + 1 - num_stage))
            num_negative = sum(1 for line in background_pathname.open())
            num_positive = int(math.ceil(positive_samples_count / (
                    (1 - min_hit_rate) * num_stages + 2)))
        start = time.time()
        if not cascade_directory.exists():
            cascade_directory.mkdir()
        opencv_traincascade_command = [
                'opencv_traincascade',
                '-data', cascade_directory,
                '-vec', vec_pathname,
                '-bg', background_pathname,
                '-numStages', num_stage,
                '-numPos', num_positive,
                '-numNeg', num_negative,
                '-featureType', feature_type,
                '-w', width,
                '-h', height,
                '-minHitRate', min_hit_rate,
                '-maxFalseAlarmRate', max_false_alarm_rate,
                '-maxWeakCount', max_weak_count,
                '-precalcValBufSize', '4000',
                '-precalcIdxBufSize', '4000',
        ]
        if num_stage == 5:
            opencv_traincascade_command.append('-random')
        opencv_traincascade_command = list(
                map(str, opencv_traincascade_command))
        print(' '.join(opencv_traincascade_command))
        subprocess.call(opencv_traincascade_command)
        new_cascade_pathname = cascade_directory / 'cascade{}.xml'.format(
                num_stage)
        shutil.copy(
                str(cascade_pathname),
                str(new_cascade_pathname))
        end = time.time()
    logging.info('elapsed time: {}'.format(end - start))


if __name__ == "__main__":
    logging.basicConfig(
            format='%(levelname)s:%(message)s', level=logging.DEBUG)
    main()
