#!/usr/bin/env python
# coding=utf-8
from __future__ import division, unicode_literals

from common import Path, rp_directory, video_suffixes, video_generator
from common import Rect
from collections import defaultdict, namedtuple
import cv2
import functools
import logging
import math
import os
import sys
import subprocess
import tempfile
import time

if sys.version_info.major == 2:
    cv2.CAP_PROP_FRAME_COUNT = 7
    cv2.LINE_AA = 16
else:
    import concurrent.futures

command_pathname = rp_directory / 'makefile/phodopus/build/main'
# command_pathname = rp_directory / 'scripts/cascade_detect/detect.py'
kitti_rate = 0.7
black = (0, 0, 0)
green = (0, 255, 0)
blue = (255, 0, 0)
red = (0, 0, 255)
yellow = (0, 255, 255)
Parameters = namedtuple(
    'Parameters',
    [
        'proto_pathname',
        'model_pathname',
        'mean_pathname',
        'cascade_model_pathname',
        'lf_proto_pathname',
        'lf_model_pathname',
        'lf_mean_pathname',
    ])


def _log_pos_samples(file_, frame_filename_str, rects):
    if len(rects) > 0:
        rects_str = ' '.join(map(str, rects))
        file_.write('{} {}\n'.format(frame_filename_str, rects_str))


def _score(x_rect, y_rect, overlap_rate):
    overlap_area = (x_rect & y_rect).area()
    union_area = x_rect.area() + y_rect.area() - overlap_area
    return overlap_area / union_area


def is_same_target(x_rect, y_rect, overlap_rate):
    return _score(x_rect, y_rect, overlap_rate) > overlap_rate


def _compare(log_rects, label_rects, overlap_rate):
    if len(log_rects) > len(label_rects):
        more_rects, less_rects = list(log_rects), list(label_rects)
    else:
        more_rects, less_rects = list(label_rects), list(log_rects)
    best_pairs = {}
    for less_rect in less_rects:
        _partial_score = functools.partial(
                _score, y_rect=less_rect, overlap_rate=overlap_rate)
        corresponding_rect = max(more_rects, key=_partial_score)
        best_pairs[less_rect] = corresponding_rect
        more_rects.remove(corresponding_rect)
    tp_pairs = [
            (best_pairs[less_rect], less_rect)
            for less_rect in best_pairs
            if is_same_target(
                    less_rect,
                    best_pairs[less_rect],
                    overlap_rate)]
    if len(log_rects) > len(label_rects):
        tp, label_tp = ([], []) if len(tp_pairs) == 0 else zip(*tp_pairs)
    else:
        label_tp, tp = ([], []) if len(tp_pairs) == 0 else zip(*tp_pairs)
    fp = [
            log_rect
            for log_rect in log_rects
            if log_rect not in tp]
    fn = [
            label_rect
            for label_rect in label_rects
            if label_rect not in label_tp]
    return tp, fp, fn


def _unwrap_or_tempfile(pathname, mode='r'):
    if pathname is None:
        return tempfile.NamedTemporaryFile(mode=mode, delete=False)
    else:
        return pathname.open(mode)


def parse(
        log_pathname,
        label_pathname,
        tp_pathname=None,
        fp_pathname=None,
        fn_pathname=None,
        overlap_rate=kitti_rate):
    log_dict = defaultdict(list)
    tp_count, fp_count, fn_count = 0, 0, 0
    log_file = log_pathname.open()
    label_file = label_pathname.open()
    tp_file = _unwrap_or_tempfile(tp_pathname, 'w')
    fp_file = _unwrap_or_tempfile(fp_pathname, 'w')
    fn_file = _unwrap_or_tempfile(fn_pathname, 'w')
    try:
        for line in log_file:
            frame_filename_str, rects_strs = line.rstrip('\n').split(' ', 1)
            log_rects = [Rect(rect_str) for rect_str in rects_strs.split()]
            log_dict[frame_filename_str] = log_rects
        for line in label_file:
            frame_filename_str, rects_strs = line.rstrip('\n').split(' ', 1)
            label_rects = [Rect(rect_str) for rect_str in rects_strs.split()]
            tp, fp, fn = _compare(
                    log_dict[frame_filename_str],
                    label_rects,
                    overlap_rate)
            _log_pos_samples(tp_file, frame_filename_str, tp)
            _log_pos_samples(fp_file, frame_filename_str, fp)
            _log_pos_samples(fn_file, frame_filename_str, fn)
            tp_count += len(tp)
            fp_count += len(fp)
            fn_count += len(fn)
    finally:
        log_file.close()
        label_file.close()
        tp_file.close()
        fp_file.close()
        fn_file.close()
    return tp_count, fp_count, fn_count


def statistics(
        tp_count,
        fp_count,
        fn_count,
        frame_count=None,
        elapsed_time=None,
        fb_measure_weight=1):
    precision = tp_count / (tp_count + fp_count)
    recall = tp_count / (tp_count + fn_count)
    fb_measure = (
            (1 + fb_measure_weight ** 2) *
            (precision * recall) /
            (fb_measure_weight ** 2 * precision + recall))
    if frame_count is None or elapsed_time is None:
        fps = None
    else:
        fps = frame_count / elapsed_time
    return precision, recall, fb_measure, fps


def evaluate_pathname(
        parameters,
        video_pathname,
        log_pathname=None,
        overlap_rate=kitti_rate):
    phodopus_evaluate_command = (
            command_pathname,
            '-v', video_pathname,
            '--proto', parameters.proto_pathname,
            '--model', parameters.model_pathname,
            '--mean', parameters.mean_pathname,
            '-c', parameters.cascade_model_pathname,
            '--lf_proto', parameters.lf_proto_pathname,
            '--lf_model', parameters.lf_model_pathname,
            '--lf_mean', parameters.lf_mean_pathname,
            '--noshow',
    )
    phodopus_evaluate_command = map(str, phodopus_evaluate_command)
    logging.debug(
            'start to use {} to evaluate {}'.format(
                    parameters.cascade_model_pathname.name,
                    video_pathname.name))
    log_file = _unwrap_or_tempfile(log_pathname, 'w')
    null_file = open(os.devnull, 'w')
    try:
        start = time.time()
        subprocess.call(
                phodopus_evaluate_command,
                stdout=log_file,
                stderr=null_file)
        end = time.time()
        elapsed_time = end - start
        if video_pathname.suffix == '':
            frame_count = len(list(video_pathname.iterdir()))
        else:
            video = cv2.VideoCapture(str(video_pathname))
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()
        logging.debug(
                'use {} to evaluate {} done'.format(
                        parameters.cascade_model_pathname.name,
                        video_pathname.name))
        log_pathname = Path(log_file.name)
        label_pathname = video_pathname.with_suffix('.txt')
        return (parse(log_pathname, label_pathname, overlap_rate=overlap_rate)
                + (frame_count, elapsed_time))
    finally:
        log_file.close()
        null_file.close()


def _evaluate_num_stage(
        testset_dir,
        cascade_model_dir,
        parameters,
        overlap_rate,
        fb_measure_weight,
        num_stage):
    logging.info(
            'starting to evaluate cascade{}.xml'.format(num_stage))
    filename = Path('{}{}{}'.format('cascade', num_stage, '.xml'))
    parameters = parameters._replace(
            cascade_model_pathname=cascade_model_dir / filename)
    tuple_results = [
            evaluate_pathname(
                    parameters,
                    pathname,
                    log_pathname=None,
                    overlap_rate=overlap_rate)
            for pathname in testset_dir.iterdir()
            if pathname.suffix in video_suffixes]
    results = zip(*tuple_results)
    return statistics(*map(sum, results), fb_measure_weight=fb_measure_weight)


def _count_num_stages(cascade_model_dir):
    return max(map(int, (
            filename_str.replace('cascade', '').replace('.xml', '')
            for filename_str in os.listdir(str(cascade_model_dir))
            if filename_str.startswith('cascade'))))


def evaluate_cascade(
        testset_dir,
        cascade_model_dir,
        parameters,
        overlap_rate=kitti_rate,
        fb_measure_weight=1):
    num_stages = _count_num_stages(cascade_model_dir)
    num_stages_range = range(
            int(math.ceil(num_stages * 2 / 3)), num_stages + 1)
    logging.info('testset_dir: {}'.format(testset_dir))
    if sys.version_info.major == 2:
        futures_dict = {
                num_stage: _evaluate_num_stage(
                        testset_dir,
                        cascade_model_dir,
                        parameters,
                        overlap_rate,
                        fb_measure_weight,
                        num_stage)
                for num_stage in num_stages_range}
        results_tuple = [
                futures_dict[key]
                for key in sorted(futures_dict)]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            futures_dict = {
                    num_stage: executor.submit(
                            _evaluate_num_stage,
                            testset_dir,
                            cascade_model_dir,
                            parameters,
                            overlap_rate,
                            fb_measure_weight,
                            num_stage)
                    for num_stage in num_stages_range}
        results_tuple = [
                futures_dict[key].result()
                for key in sorted(futures_dict)]
    precisions, recalls, fb_measures, fpses = zip(*results_tuple)
    precisions = ['{:.3}'.format(precision) for precision in precisions]
    recalls = ['{:.3}'.format(recall) for recall in recalls]
    fb_measures = [
            '{:.3}'.format(fb_measure)
            for fb_measure in fb_measures]
    fps = ['{:.3f}'.format(fps) for fps in fpses]
    log_dir = cascade_model_dir
    log_pathname = log_dir / 'evaluate.log'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(str(log_pathname), 'w')
    logger.addHandler(file_handler)
    logger.info('overlap_rate: {:2}'.format(overlap_rate))
    logger.info('range: {}'.format(num_stages_range))
    logger.info('precisions: {}'.format(precisions))
    logger.info('recalls: {}'.format(recalls))
    logger.info('fb_measure_weight: {}, fb_measure: {}'.format(
        fb_measure_weight, fb_measures))
    logger.info('fps: {}'.format(fps))


def _file2dict(pathname):
    dict_ = defaultdict(list)
    if pathname is None:
        return dict_
    else:
        with pathname.open() as file_:
            for line in file_:
                frame_filename_str, rects_strs = line.rstrip(
                        '\n').split(' ', 1)
                rects_str = rects_strs.split()
                dict_[frame_filename_str] = [
                        Rect(rect_str)
                        for rect_str in rects_str]
            return dict_


def _draw_dict(frame, frame_filename_str, pos_sample_dict, color):
    for rect in pos_sample_dict[frame_filename_str]:
        cv2.rectangle(
                frame,
                (rect.x, rect.y),
                (rect.x + rect.width, rect.y + rect.height),
                color,
                thickness=2)


def _draw_legend(frame, frame_filename_str):
    cv2.putText(
            frame, 'frame_filename: {}'.format(frame_filename_str),
            (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, black, 2, cv2.LINE_AA)
    cv2.putText(
            frame, 'label: green',
            (0, 40), cv2.FONT_HERSHEY_PLAIN, 1, green, 2, cv2.LINE_AA)
    cv2.putText(
            frame, 'true_positive: blue',
            (0, 60), cv2.FONT_HERSHEY_PLAIN, 1, blue, 2, cv2.LINE_AA)
    cv2.putText(
            frame, 'false_positive: red',
            (0, 80), cv2.FONT_HERSHEY_PLAIN, 1, red, 2, cv2.LINE_AA)
    cv2.putText(
            frame, 'false_negative: yellow',
            (0, 100), cv2.FONT_HERSHEY_PLAIN, 1, yellow, 2, cv2.LINE_AA)


def watch(
            sample_pathname,
            label_pathname,
            tp_pathname=None,
            fp_pathname=None,
            fn_pathname=None,
            is_sync=True,
            is_evaluate_save=False,
            delay=300):
    label_dict = _file2dict(label_pathname)
    tp_dict = _file2dict(tp_pathname)
    fp_dict = _file2dict(fp_pathname)
    fn_dict = _file2dict(fn_pathname)
    cv2.namedWindow('watch')
    video = video_generator(sample_pathname)
    if is_evaluate_save:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output = cv2.VideoWriter(
                '{}-evaluate{}'.format(
                        sample_pathname.stem,
                        '.avi'),
                fourcc, 3.0, (1280, 720))
    for frame_filename_str, frame in video:
        if frame_filename_str not in label_dict and is_sync:
            continue
        else:
            _draw_dict(frame, frame_filename_str, label_dict, green)
            _draw_dict(frame, frame_filename_str, tp_dict, blue)
            _draw_dict(frame, frame_filename_str, fp_dict, red)
            _draw_dict(frame, frame_filename_str, fn_dict, yellow)
            _draw_legend(frame, frame_filename_str)
            if is_evaluate_save:
                output.write(frame)
            cv2.imshow('watch', frame)
            cv2.waitKey(delay)
    if is_evaluate_save:
        output.release()
    cv2.destroyAllWindows()
