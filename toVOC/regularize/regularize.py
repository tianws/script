#!/usr/bin/env python
# coding=utf-8
from __future__ import division, unicode_literals

from common import find_sample, Path, video_suffixes
from chardet.universaldetector import UniversalDetector
from collections import defaultdict
import cv2
from functools import partial
import logging
import os
from PIL import Image
import shutil
import subprocess
import tempfile


def backup(label_pathname, backup_pathname):
    shutil.copy(str(label_pathname), str(backup_pathname))


def convert(rects_str):
    return [[int(value_str)
            for value_str in rect_str.split(',')]
            for rect_str in rects_str.split()]


def wrapper(label_pathname, function):
    with tempfile.NamedTemporaryFile('r+') as tmp_file, \
            label_pathname.open() as label_file:
        function(label_file, tmp_file)
        tmp_file.seek(0)
        shutil.copy(tmp_file.name, str(label_pathname))


def dos2unix(label_file, tmp_file):
    for line in label_file:
        line.replace('\r\n', '\n')
        tmp_file.write(line)


def encoding2utf8(label_file, tmp_file):
    detector = UniversalDetector()
    for line in label_file:
        detector.feed(line)
        if detector.done:
            break
    detector.close()
    encoding = detector.result['encoding']
    for line in label_file:
        new_line = line.decode(encoding).encode()
        tmp_file.write(new_line)


def is_line_valid(line, sample_pathname):
    null_file = open(os.devnull, 'w')
    if len(line.split()) == 1:
        logging.warning('no rects!')
        return False
    frame_filename_str, rects_str = line.rstrip('\n').split(' ', 1)
    if sample_pathname.suffix in list(video_suffixes).remove(''):
        video = cv2.VideoCapture(str(sample_pathname))
        image_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        image_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        video.release()
    else:
        image_pathname = sample_pathname / frame_filename_str
        if not image_pathname.exists():
            logging.warning('{} does not exist!'.format(image_pathname))
            return False
        try:
            image = Image.open(str(image_pathname))
        except IOError:
            logging.warning('image does not exist!')
            return False
        check_command = ['jpeginfo', '-c', str(image_pathname)]
        # So you should install jpeginfo in computer firstly.
        if subprocess.call(check_command, stdout=null_file) != 0:
            logging.warning('image is broken!')
            return False
        image_width, image_height = image.size
        if image_width * image_height == 0:
            logging.warning('image is empty!')
            return False
    rects = convert(rects_str)
    for rect in rects:
        if len(rect) % 4 != 0:
            logging.warning('rect is not module by 4! {}'.format(rect))
            return False
        x, y, width, height = rect
        if width <= 0 or height <= 0:
            logging.warning('rect\'s width or height is unvalid!')
            return False
        if (x >= image_width or y >= image_height or
                x + width > image_width or y + height > image_height):
            logging.warning('rect\'s point coordinates is invalid!')
            return False
    return True


def remove_unvalid_line(label_file, tmp_file, sample_pathname):
    for index, line in enumerate(label_file, start=1):
        if is_line_valid(line, sample_pathname):
            tmp_file.write(line)
        else:
            logging.warning(
                    '{} {}: {}'.format(
                            label_file.name, index, line.rstrip('\n')))


def uniq(label_file, tmp_file):
    uniq_command = ['uniq', label_file.name]
    subprocess.call(uniq_command, stdout=tmp_file)
    logging.info('{} uniq done'.format(label_file.name))


def remove_duplicated_lines(label_file, tmp_file):
    label_dict = defaultdict(list)
    for line in label_file:
        frame_filename_str, rects_str = line.rstrip('\n').split(' ', 1)
        label_dict[frame_filename_str].append(rects_str)
    for frame_filename_str, rects_str_list in label_dict.items():
        if len(rects_str_list) == 1:
            line = '{} {}\n'.format(frame_filename_str, rects_str_list[0])
            tmp_file.write(line)
        else:
            logging.warning(
                    '{} detect duplicated: {} {}'.format(
                            label_file.name,
                            frame_filename_str,
                            rects_str_list))


def sort(label_file, tmp_file):
    label_dict = {}
    for line in label_file:
        frame_filename_str = line.rstrip('\n').split(' ')[0]
        try:
            frame_number = int(frame_filename_str.split('.')[0])
        except ValueError:
            shutil.copy(label_file.name, tmp_file.name)
            return
        label_dict[frame_number] = line
    for frame_number in sorted(label_dict.keys()):
        tmp_file.write(label_dict[frame_number])


def main():
    trainset_directory = Path('/home/acgtyrant/BigDatas/car/trainset')
    backup_directory = trainset_directory / 'backup'
    try:
        backup_directory.mkdir()
    except Exception as e:
        logging.warning(e, exc_info=True)
    label_pathnames = [
            pathname
            for pathname in trainset_directory.iterdir()
            if pathname.suffix == '.txt']
    for label_pathname in label_pathnames:
        backup(label_pathname, backup_directory / label_pathname.name)
    for label_pathname in label_pathnames:
        try:
            sample_pathname = find_sample(label_pathname)
        except:
            logging.warning(
                    '{}\'s sample does not exist'.format(label_pathname))
        else:
            practical_remove_unvalid_line = partial(
                    remove_unvalid_line,
                    sample_pathname=sample_pathname)
            functions = [
                    dos2unix,
                    sort,
                    uniq,
                    practical_remove_unvalid_line,
                    remove_duplicated_lines,
                    sort]
            for function in functions:
                wrapper(label_pathname, function)
        logging.info('regularize {} done'.format(label_pathname))


if __name__ == '__main__':
    main()
