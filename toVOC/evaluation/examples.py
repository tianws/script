#!/usr/bin/env python
# coding=utf-8
from __future__ import division, unicode_literals

from common import video_suffixes
from common import Path
import itertools
import logging
import os
import phodopus
import squirrel


def evaluate():
    logging.basicConfig(
            format='%(levelname)s:%(message)s', level=logging.DEBUG)
    home_dir = Path(os.path.expanduser('~'))
    hfm_dir = Path(
            '/home/acgtyrant/Projects/HamsterForMTK/'
            'Hamster_Android_SDK/src/com/mapbar/hamster')
    testset_dir = home_dir / 'BigDatas/car/testset'
    parameters = phodopus.Parameters(
            hfm_dir / 'deploy.prototxt',
            hfm_dir / 'mb_confirm__iter_60000.caffemodel',
            hfm_dir / 'net_mean_file',
            Path('/home/acgtyrant/BigDatas/car/cascade/cascade15.xml'),
            hfm_dir / 'location_finer_1026_1_test.prototxt',
            hfm_dir / 'lf1__iter_60000.caffemodel',
            hfm_dir / 'mean_file')
    qin_rate = 0.5
    # KITTI 的标准是 70%, 即 Phodopus.kitti_rate, 但秦曰：数据可能会很难看，所以就用 50% 吧
    tp_counts, fp_counts, fn_counts = 0, 0, 0
    for pathname in testset_dir.iterdir():
        if pathname.suffix in video_suffixes:
            log_pathname = pathname.with_suffix('.log')
            label_pathname = pathname.with_suffix('.txt')
            logging.basicConfig(
                    format='%(levelname)s:%(message)s', level=logging.INFO)
            phodopus.evaluate_pathname(
                    parameters,
                    pathname,
                    log_pathname,
                    qin_rate)
            tp_count, fp_count, fn_count = phodopus.parse(
                    log_pathname,
                    label_pathname,
                    overlap_rate=0.5)
            tp_counts += tp_count
            fp_counts += fp_count
            fn_counts += fn_count
    precision, recall, fb_measure, _ = phodopus.statistics(
            tp_counts,
            fp_counts,
            fn_counts)
    logging.info('precision: {:.3}'.format(precision))
    logging.info('recall: {:.3}'.format(recall))
    logging.info('fb_measure: {:.3}'.format(fb_measure))


def evaluate_phodopus():
    home_dir = Path(os.path.expanduser('~'))
    hfm_dir = Path(
        '/home/acgtyrant/Projects/HamsterForMTK/'
        'Hamster_Android_SDK/src/com/mapbar/hamster')
    testset_dir = home_dir / 'BigDatas/car/testset'
    num_stages = (14, 15, 16)
    ratios = (0.25, 0.5, 0.75, 1)
    for num_stage, ratio in itertools.product(num_stages, ratios):
        cascade_model_dir = Path(
                '/home/acgtyrant/BigDatas/car/scaled_cascade-{}-{}'.format(
                        num_stage, ratio))
        parameters = phodopus.Parameters(
                hfm_dir / 'deploy.prototxt',
                hfm_dir / 'mb_confirm__iter_60000.caffemodel',
                hfm_dir / 'net_mean_file',
                None,
                hfm_dir / 'location_finer_1026_1_test.prototxt',
                hfm_dir / 'lf1__iter_60000.caffemodel',
                hfm_dir / 'mean_file')
        qin_rate = 0.5
        # KITTI 的标准是 70%, 即 Phodopus.kitti_rate, 但秦曰：数据可能会很难看，所以就用 50% 吧
        logging.basicConfig(
                format='%(levelname)s:%(message)s', level=logging.DEBUG)
        phodopus.evaluate_cascade(
                testset_dir,
                cascade_model_dir,
                parameters,
                qin_rate)


def watch_phodopus():
    home_dir = Path(os.path.expanduser('~'))
    testset_dir = home_dir / 'BigDatas/daytime/trainset'
    video_pathname = testset_dir / '1362384843.mov'
    label_pathname = video_pathname.with_suffix('.txt')
    log_pathname = video_pathname.with_suffix('.log')
    tp_pathname = video_pathname.with_suffix('.tp')
    fp_pathname = video_pathname.with_suffix('.fp')
    fn_pathname = video_pathname.with_suffix('.fn')
    tp_count, fp_count, fn_count = phodopus.parse(
            log_pathname,
            label_pathname,
            tp_pathname,
            fp_pathname,
            fn_pathname)
    precision, recall, fb_measure, _ = phodopus.statistics(
            tp_count,
            fp_count,
            fn_count)
    print('precision: {:.3}'.format(precision))
    print('recall: {:.3}'.format(recall))
    print('fb_measure: {:.3}'.format(fb_measure))
    phodopus.watch(
            video_pathname,
            label_pathname,
            tp_pathname,
            fp_pathname,
            fn_pathname)


def day_task():
    home_dir = Path(os.path.expanduser('~'))
    testset_dir = home_dir / 'BigDatas/daytime/testset'
    tp_counts, fp_counts, fn_counts = 0, 0, 0
    for video_pathname in [
            pathname
            for pathname in testset_dir.iterdir()
            if pathname.suffix == '.avi']:
        label_pathname = video_pathname.with_suffix('.txt')
        log_pathname = video_pathname.with_suffix('.log')
        tp_pathname = video_pathname.with_suffix('.tp')
        fp_pathname = video_pathname.with_suffix('.fp')
        fn_pathname = video_pathname.with_suffix('.fn')
        tp_count, fp_count, fn_count = phodopus.parse(
                log_pathname,
                label_pathname,
                tp_pathname,
                fp_pathname,
                fn_pathname)
        tp_counts += tp_count
        fp_counts += fp_count
        fn_counts += fn_count
    precision, recall, fb_measure, _ = phodopus.statistics(
            tp_counts,
            fp_counts,
            fn_counts)
    print('precision: {:.3}'.format(precision))
    print('recall: {:.3}'.format(recall))
    print('fb_measure: {:.3}'.format(fb_measure))


def evaluate_squirrel():
    home_dir = Path(os.path.expanduser('~'))
    testset_dir = home_dir / 'BigDatas/squirrel/testset'
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    squirrel.evaluate(testset_dir)


def watch_squirrel():
    logging.basicConfig(
            format='%(levelname)s:%(message)s', level=logging.DEBUG)
    home_dir = Path(os.path.expanduser('~'))
    testset_dir = home_dir / 'BigDatas/squirrel/testset'
    video_pathname = testset_dir / '1970-1-2_08-40-06.avi'
    label_pathname = video_pathname.with_suffix('.txt')
    tp_pathname = video_pathname.with_suffix('.tp')
    fp_pathname = video_pathname.with_suffix('.fp')
    fn_pathname = video_pathname.with_suffix('.fn')
    # tp_count, fp_count, fn_count = squirrel.parse(
            # log_pathname,
            # label_pathname,
            # tp_pathname,
            # fp_pathname,
            # fn_pathname)
    # precision, recall, fb_measure, _ = squirrel.statistics(
            # tp_count,
            # fp_count,
            # fn_count)
    # print('precision: {:.3}'.format(precision))
    # print('recall: {:.3}'.format(recall))
    # print('fb_measure: {:.3}'.format(fb_measure))
    squirrel.watch(
            video_pathname,
            label_pathname,
            tp_pathname,
            fp_pathname,
            fn_pathname,
            is_evaluate_save=True)


# evaluate()
evaluate_phodopus()
# evaluate_squirrel()
# watch_squirrel()
# watch_squirrel()
