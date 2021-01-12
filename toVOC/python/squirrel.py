#!/usr/bin/env python
# coding=utf-8
from __future__ import division, unicode_literals

from common import Path
from common import Point
from common import rp_directory, video_suffixes, video_generator
from collections import defaultdict, OrderedDict
import cv2
import itertools
import logging
import numpy
import os
import subprocess
import sys
import tempfile
import time

if sys.version_info.major == 2:
    cv2.CAP_PROP_FRAME_COUNT = 7
    cv2.LINE_AA = 16
else:
    import concurrent.futures

command_pathname = rp_directory / 'makefile/squirrel/build/main'
kitti_rate = 0.7
black = (0, 0, 0)
green = (0, 255, 0)
blue = (255, 0, 0)
red = (0, 0, 255)
yellow = (0, 255, 255)
squirrel_weight = 40


zero_point = Point(0, 0)


def _log_pos_samples(file_, frame_filename_str, lanes):
    for lane in lanes:
        strs = ' '.join(map(str, lane))
        if len(strs) > 0:
            file_.write('{} {}\n'.format(frame_filename_str, strs))


def loss(log_lane, label_lane):
    degree = min(len(log_lane), len(label_lane)) - 1
    log_x, log_y = zip(*map(tuple, log_lane))
    label_x, label_y = zip(*map(tuple, label_lane))
    log_z = numpy.polyfit(log_y, log_x, degree)
    label_z = numpy.polyfit(label_y, label_x, degree)
    log_p = numpy.poly1d(log_z)
    label_p = numpy.poly1d(label_z)
    all_y = log_y + label_y
    return abs(log_p(all_y) - label_p(all_y))


def area_loss(log_lane, label_lane):
    degree = min(len(log_lane), len(label_lane)) - 1
    log_x, log_y = zip(*map(tuple, log_lane))
    label_x, label_y = zip(*map(tuple, label_lane))
    log_z = numpy.polyfit(log_y, log_x, degree)
    label_z = numpy.polyfit(label_y, label_x, degree)
    log_p = numpy.poly1d(log_z)
    label_p = numpy.poly1d(label_z)
    all_y = range(min(log_y), max(log_y), 50)
    return sum(abs(log_p(all_y) - label_p(all_y)) * 50)


def _is_same_target(log_lane, label_lane, threshold, loss_method):
    result = loss_method(log_lane, label_lane)
    if loss_method is loss:
        return all(result < threshold)
    if loss_method is area_loss:
        log_x, log_y = zip(*map(tuple, log_lane))
        lane_len = max(log_y) - min(log_y)
        return result < lane_len * threshold


def _compare(log_lanes, label_lanes, threshold, loss_method):
    public_len = min(len(log_lanes), len(label_lanes))
    if public_len == 0:
        return [], log_lanes, label_lanes, float("inf")
    if len(log_lanes) > len(label_lanes):
        more_lanes, less_lanes = log_lanes, label_lanes
    else:
        more_lanes, less_lanes = label_lanes, log_lanes
    log_lanes_permutations = [
            tuple(permutation[:public_len])
            for permutation in itertools.permutations(more_lanes)]
    permutations_result = {}
    for permutation in log_lanes_permutations:
        losses = sum([
                loss_method(more_lane, less_lanes[index])
                for index, more_lane in enumerate(permutation)])
        if loss_method == loss:
            losses = tuple(losses)
        permutations_result[losses] = permutation
    if loss_method == loss:
        min_losses = min(permutations_result.keys(), key=sum)
    elif loss_method == area_loss:
        min_losses = min(permutations_result.keys())
    best_permutation = permutations_result[min_losses]
    result = [
            (more_lane, less_lanes[index])
            for index, more_lane in enumerate(best_permutation)
            if _is_same_target(
                    more_lane,
                    less_lanes[index],
                    threshold,
                    loss_method)]
    if len(log_lanes) > len(label_lanes):
        tp, label_tp = ([], []) if len(result) == 0 else zip(*result)
    else:
        label_tp, tp = ([], []) if len(result) == 0 else zip(*result)
    fp = [
            log_lane
            for log_lane in log_lanes
            if log_lane not in tp]
    fn = [
            label_lane
            for label_lane in label_lanes
            if label_lane not in label_tp]
    return tp, fp, fn, min_losses


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
        threshold=squirrel_weight,
        loss_method=loss):
    log_dict = defaultdict(list)
    label_dict = defaultdict(list)
    tp_count, fp_count, fn_count = 0, 0, 0
    log_file = log_pathname.open()
    label_file = label_pathname.open()
    tp_file = _unwrap_or_tempfile(tp_pathname, 'w')
    fp_file = _unwrap_or_tempfile(fp_pathname, 'w')
    fn_file = _unwrap_or_tempfile(fn_pathname, 'w')
    try:
        for line in log_file:
            frame_filename_str, points_strs = line.rstrip('\n').split(' ', 1)
            log_lane = [
                    Point(point_str)
                    for point_str in points_strs.split()]
            if all([point != zero_point for point in log_lane]):
                log_dict[frame_filename_str].append(log_lane)
        for line in label_file:
            frame_filename_str, points_strs = line.rstrip('\n').split(' ', 1)
            label_lane = [
                    Point(point_str)
                    for point_str in points_strs.split()]
            label_dict[frame_filename_str].append(label_lane)
        try:
            label_dict = OrderedDict(sorted(
                    label_dict.items(),
                    key=lambda key: int(key[0].split('.')[0])))
        except ValueError:
            pass
        for frame_filename_str in label_dict:
            tp, fp, fn, losses = _compare(
                    log_dict[frame_filename_str],
                    label_dict[frame_filename_str],
                    threshold,
                    loss_method)
            logging.info('{}: {}'.format(frame_filename_str, losses))
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


def _evaluate_pathname(video_pathname, log_pathname, trheshold):
    squirrel_evaluate_command = (
            command_pathname,
            '-v', video_pathname,
            '--noshow',  # 本脚本原则上不显示视频，旨在加快评测效率
    )
    squirrel_evaluate_command = map(str, squirrel_evaluate_command)
    logging.debug('start to evaluate {}'.format(video_pathname.name))
    log_file = _unwrap_or_tempfile(log_pathname, 'w')
    null_file = open(os.devnull, 'w')
    try:
        start = time.time()
        subprocess.call(
                squirrel_evaluate_command,
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
        logging.debug('evaluate {} done'.format(video_pathname.name))
        log_pathname = Path(log_file.name)
        label_pathname = video_pathname.with_suffix('.txt')
        tp_pathname = video_pathname.with_suffix('.tp')
        fp_pathname = video_pathname.with_suffix('.fp')
        fn_pathname = video_pathname.with_suffix('.fn')
        tp_count, fp_count, fn_count = parse(
                log_pathname,
                label_pathname,
                tp_pathname,
                fp_pathname,
                fn_pathname,
                trheshold)
        return statistics(
                tp_count,
                fp_count,
                fn_count,
                frame_count,
                elapsed_time)
    finally:
        log_file.close()
        null_file.close()


def evaluate(
        testset_dir,
        threshold=squirrel_weight,
        fb_measure_weight=1):
    logging.info('testset_dir: {}'.format(testset_dir))
    if sys.version_info.major == 2:
        results = [
                _evaluate_pathname(
                        pathname,
                        pathname.with_suffix('.log'),
                        threshold)
                for pathname in testset_dir.iterdir()
                if pathname.suffix in video_suffixes]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            results = [
                    executor.submit(
                            _evaluate_pathname,
                            pathname,
                            pathname.with_suffix('.log'),
                            threshold).result()
                    for pathname in testset_dir.iterdir()
                    if pathname.suffix in video_suffixes]
    precision, recall, fb_measure, fps = map(numpy.mean, zip(*results))
    logging.info('threshold: {}'.format(threshold))
    logging.info('precision: {:.3}'.format(precision))
    logging.info('recall: {:.3}'.format(recall))
    logging.info('fb_measure_weight: {}, fb_measure: {:.3}'.format(
            fb_measure_weight, fb_measure))
    logging.info('fps: {:.3}'.format(fps))


def _file2dict(pathname):
    dict_ = defaultdict(list)
    if pathname is None:
        return dict_
    else:
        with pathname.open() as file_:
            for line in file_:
                frame_filename_str, points_strs = line.rstrip(
                        '\n').split(' ', 1)
                points_str = points_strs.split()
                dict_[frame_filename_str].append([
                        Point(point_str)
                        for point_str in points_str])
            return dict_


def _draw_dict(frame, frame_filename_str, pos_sample_dict, color):
    for lane in pos_sample_dict[frame_filename_str]:
        x, y = zip(*map(tuple, lane))
        degree = len(x) - 1
        if degree > 1:
            z = numpy.polyfit(y, x, degree)
            p = numpy.poly1d(z)
            height = frame.shape[0]
            y = range(height // 2, height, 10)
            x = p(y)
            lane = [Point(x, y) for x, y in zip(x, y)]
        for left_point, right_point in zip(lane[:-1], lane[1:]):
            cv2.line(frame, tuple(left_point), tuple(right_point), color, 2)
    del pos_sample_dict[frame_filename_str]


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
    cv2.destroyAllWindows()
