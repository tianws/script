#!/usr/bin/env python
# coding=utf-8
from __future__ import division, unicode_literals

from common import Path
import math
import logging
import os
import subprocess
import shutil


def main():
    directory = Path(os.path.expanduser('~')) / 'BigDatas/car'
    cascade_pathname = directory / 'cascade'
    info_pathname = directory / 'positive_samples.txt'
    weight, height = 20, 20
    vec_pathname = directory / '{}_{}_{}'.format(
            weight,
            height,
            'positive_samples.vec')
    background_pathname = directory / 'positive_samples.txt'
    num_stages = 20
    min_hit_rate = 0.99
    max_false_alarm_rate = 0.5
    positive_samples_count = sum(map(int, [
            line.split(' ', 2)[1] for line in info_pathname.open()]))
    num_positive = int(math.ceil(positive_samples_count / (
            (1 - min_hit_rate) * num_stages + 2)))
    num_negative = num_positive
    feature_type = 'LBP'
    max_weak_count = 1000
    if not cascade_pathname.exists():
        cascade_pathname.mkdir()
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
    cascade_xml_pathname = cascade_pathname / 'cascade.xml'
    for num_stage in range(num_stages, num_stages * 2 // 3 - 1, -1):
        opencv_traincascade_command = [
                'opencv_traincascade',
                '-data', cascade_pathname,
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
        logging.info(' '.join(opencv_traincascade_command))
        subprocess.call(opencv_traincascade_command)
        new_cascade_xml_pathname = Path('{}/{}{}{}'.format(
                cascade_pathname, 'cascade', num_stage, '.xml'))
        shutil.copy(str(cascade_xml_pathname), str(new_cascade_xml_pathname))
    cascade_xml_pathname.unlink()


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    main()
