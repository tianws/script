#!/usr/bin/env python
# coding=utf-8
from __future__ import division, unicode_literals

import cv2
import os
import sys

if sys.version_info.major == 2:
    from pathlib2 import Path
else:
    from pathlib import Path
    unicode = str
assert Path

caffe_directory = Path(os.getenv('CAFFE_DIR'))
rp_directory = Path(os.getenv('RP_DIR'))
video_suffixes = ('.mov', '.avi', '.mp4', '')


def video_generator(video_pathname):
    if video_pathname.suffix == '':
        # 可以把一个图片目录包装成视频来读取帧编号和帧。
        image_pathnames = list(video_pathname.iterdir())
        if (Path('{}/20.jpg'.format(video_pathname)).exists()):
            # 样本图片目录的帧编号同时有三个格式：0, 20, 40..., 20, 41, 62...
            # 和毫无规律的。于是先判定是否为前两种格式，若是，则按帧编号排序
            image_pathnames = sorted(
                    image_pathnames, key=lambda x: int(x.name.split('.')[0]))
        for image_pathname in image_pathnames:
            yield str(image_pathname.name), cv2.imread(str(image_pathname))
        raise StopIteration
    else:
        index = 0
        video = cv2.VideoCapture(str(video_pathname))
        while True:
            return_value, frame = video.read()
            if not return_value:
                video.release()
                raise StopIteration
            else:
                yield '{}.jpg'.format(index), frame
            index += 1


def find_sample(label_pathname):
    for pathname in [
            label_pathname.with_suffix(suffix)
            for suffix in video_suffixes]:
        if pathname.exists():
            return pathname
    raise Exception('Sample does not exist.')


class Point():
    def __init__(self, *args):
        if len(args) == 2:
            self.x, self.y = args
        elif len(args) == 1 and type(args[0]) in (str, unicode):
            self.__init__(*[int(string) for string in args[0].split(',')])
        else:
            raise ValueError(
                    '{} is unvalid to construct a Point object'.format(args))

    def __str__(self):
        return '{},{}'.format(self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self == other

    def __len__(self):
        return 2

    def __getitem__(self, key):
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        else:
            raise IndexError


class Rect():

    def __init__(self, *args):
        if len(args) == 4:
            self.x, self.y, self.width, self.height = args
        elif len(args) == 1 and type(args[0]) in (str, unicode):
            self.__init__(*[int(string) for string in args[0].split(',')])
        else:
            raise ValueError(
                    '{} is unvalid to construct a Rect object'.format(args))

    def __str__(self):
        return '{},{},{},{}'.format(self.x, self.y, self.width, self.height)

    def __and__(self, rect):
        x = max(self.x, rect.x)
        y = max(self.y, rect.y)
        width = min(self.x + self.width, rect.x + rect.width) - x
        height = min(self.y + self.height, rect.y + rect.height) - y
        if width <= 0 or height <= 0:
            return Rect(0, 0, 0, 0)
        else:
            return Rect(x, y, width, height)

    def area(self):
        return self.width * self.height

    def __len__(self):
        return 4

    def __getitem__(self, key):
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        elif key == 2:
            return self.width
        elif key == 3:
            return self.height
        else:
            raise IndexError
