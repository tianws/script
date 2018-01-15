#!/usr/bin/env python
# coding=utf-8
from __future__ import division, unicode_literals

from common import find_sample, video_generator
from common import Path
import cv2
import logging


def _handle_video(label_pathname, video_pathname, samples_directory):
    frame_filename_strs = []
    for line in label_pathname.open():
        frame_filename_str, rects_str = line.rstrip('\n').split(' ', 1)
        frame_filename_strs.append(frame_filename_str)
    video = video_generator(video_pathname)
    for frame_filename_str, frame in video:
        if frame_filename_str in frame_filename_strs:
            frame_pathname_str = '{}/{}-{}'.format(
                    samples_directory,
                    video_pathname.stem,
                    frame_filename_str)
            cv2.imwrite(frame_pathname_str, frame)
            logging.info('{} done'.format(frame_pathname_str))


def _handle_info(sample_pathname, info_pathname):
    label_pathname = sample_pathname.with_suffix('.txt')
    for line in label_pathname.open():
        frame_filename_str, rects_str = line.rstrip('\n').split(' ', 1)
        new_rects_str = rects_str.replace(',', ' ')
        new_line = 'samples/{}-{} {} {}\n'.format(
                sample_pathname.stem,
                frame_filename_str,
                len(rects_str.split()),
                new_rects_str)
        with info_pathname.open('a') as info_file:
            info_file.write(new_line)


def main():
    directory = Path('/home/acgtyrant/BigDatas/car')
    samples_directory = directory / 'samples'
    if not samples_directory.exists():
        samples_directory.mkdir()
    trainset_directory = directory / 'trainset'
    info_pathname = directory / 'positive_samples.txt'
    for label_pathname in [
            pathname
            for pathname in trainset_directory.iterdir()
            if pathname.suffix == '.txt']:
        try:
            sample_pathname = find_sample(label_pathname)
            _handle_video(
                    label_pathname,
                    sample_pathname,
                    samples_directory)
            _handle_info(sample_pathname, info_pathname)
        except Exception:
            samples_directory.rmdir()
            info_pathname.unlink()
            raise
        logging.info('handle {} done'.format(label_pathname))


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    main()
