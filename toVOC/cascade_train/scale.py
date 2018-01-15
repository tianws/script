#!/usr/bin/env python
# coding=utf-8
from __future__ import division, unicode_literals

from common import Rect
from common import Path
import cv2
import os


def main():
    directory = Path(os.path.expanduser('~')) / 'BigDatas/car'
    info_pathname = directory / 'positive_samples.txt'
    new_info_pathname = directory / 'scaled_positive_samples.txt'
    with info_pathname.open() as info_file, \
            new_info_pathname.open('w') as new_info_file:
        for line in info_file:
            frame_filename_str, num_str, rects_str = (
                    line.rstrip('\n').split(' ', 2))
            image_pathname = directory / frame_filename_str
            image = cv2.imread(str(image_pathname))
            image_height, image_width, _ = image.shape
            rects_strs = rects_str.split()
            rects = []
            for index in range(0, len(rects_strs), 4):
                x, y, width, height = map(int, rects_strs[index:index + 4])
                new_width = int(width * 1.2)
                new_x = int(x - width * 0.1)
                new_height = int(height * 1.2)
                new_y = int(y - height * 0.1)
                if (new_x >= 0 and new_y >= 0
                        and new_width < image_width
                        and new_height < image_height):
                    rects.append(Rect(new_x, new_y, new_width, new_height))
                else:
                    rects.append(Rect(x, y, width, height))
            new_rects_str = ' '.join([
                    ' '.join(map(str, tuple(rect)))
                    for rect in rects])
            new_info_file.write('{} {} {}\n'.format(
                    frame_filename_str,
                    num_str,
                    new_rects_str))


if __name__ == '__main__':
    main()
