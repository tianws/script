#!/usr/bin/env python
# coding=utf-8
from __future__ import division, unicode_literals

from common import Path
import math
import logging
import itertools
import os
import subprocess
import shutil


def main():
    directory = Path(os.path.expanduser('~')) / 'BigDatas/car'
    cascade_directory = directory / 'cascade'
    info_pathname = directory / 'scaled_positive_samples.txt'
    weight, height = 20, 20
    vec_pathname = directory / 'scaled_{}_{}_{}'.format(
            weight,
            height,
            'positive_samples.vec')
    background_pathname = directory / 'positive_samples.txt'
    num_stageses = (14, 15, 16)
    min_hit_rate = 0.99
    max_false_alarm_rate = 0.5
    positive_samples_count = sum(map(int, [
            line.split(' ', 2)[1] for line in info_pathname.open()]))
    ratios = (1, 0.75, 0.5, 0.25)
    feature_type = 'LBP'
    max_weak_count = 1000
    if not vec_pathname.exists():
        opencv_createsamples_command = [
                'opencv_createsamples',
                '-info', info_pathname,
                '-vec', vec_pathname,
                '-num', positive_samples_count,
                '-w', weight,
                '-h', height,
        ]
        opencv_createsamples_command = list(
                map(str, opencv_createsamples_command))
        logging.info(' '.join(opencv_createsamples_command))
        subprocess.call(opencv_createsamples_command)
    cascade_xml_pathname = cascade_directory / 'cascade.xml'
    for num_stages, ratio in itertools.product(num_stageses, ratios):
        if not cascade_directory.exists():
            cascade_directory.mkdir()
        num_positive = int(math.ceil(positive_samples_count / (
                (1 - min_hit_rate) * num_stages + 2)))
        num_negative = int(num_positive * ratio)
        for num_stage in range(num_stages, num_stages * 2 // 3 - 1, -1):
            opencv_traincascade_command = [
                    'opencv_traincascade',
                    '-data', cascade_directory,
                    '-vec', vec_pathname,
                    '-bg', background_pathname,
                    '-numStages', num_stage,
                    '-numPos', num_positive,
                    '-numNeg', num_negative,
                    '-featureType', feature_type,
                    '-w', weight,
                    '-h', height,
                    '-minHitRate', min_hit_rate,
                    '-maxFalseAlarmRate', max_false_alarm_rate,
                    '-maxWeakCount', max_weak_count,
                    '-precalcValBufSize', '4000',
                '-precalcIdxBufSize', '4000',
                '-random',
            ]
            opencv_traincascade_command = list(
                    map(str, opencv_traincascade_command))
            log_dir = cascade_directory
            log_pathname = log_dir / 'train_command.log'
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            file_handler = logging.FileHandler(str(log_pathname), 'w')
            logger.addHandler(file_handler)
            logger.info(' '.join(opencv_traincascade_command))
            subprocess.call(opencv_traincascade_command)
            new_cascade_xml_pathname = Path('{}/{}{}{}'.format(
                    cascade_directory, 'cascade', num_stage, '.xml'))
            shutil.move(
                    str(cascade_xml_pathname), str(new_cascade_xml_pathname))
        new_cascade_directory = directory / 'scaled_cascade-{}-{}'.format(
                num_stages, ratio)
        shutil.move(str(cascade_directory), str(new_cascade_directory))


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    main()
