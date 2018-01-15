#!/usr/bin/env python
# coding=utf-8 from __future__ import division, unicode_literals

import argparse
from common import video_generator, Path, Rect
import cv2
import sys


def main():
    parser = argparse.ArgumentParser(
            description='读取 cascade_model, 并直接调用 detectMultiScale 做全图多目标识别')
    parser.add_argument('-v', action='store', dest='video_pathname_str')
    parser.add_argument('-c', action='store', dest='cascade_pathname_str')
    parser.add_argument('--noshow', action='store_true', default=False)
    special_help = '无用参数，仅为兼容 makefile/phodopus 现有的 parameter.'
    parser.add_argument('--proto', action='store', help=special_help)
    parser.add_argument('--model', action='store', help=special_help)
    parser.add_argument('--mean', action='store', help=special_help)
    parser.add_argument('--lf_proto', action='store', help=special_help)
    parser.add_argument('--lf_model', action='store', help=special_help)
    parser.add_argument('--lf_mean', action='store', help=special_help)
    # 以上六行的参数目前对脚本没用，仅仅为了兼容 makefile/phodopus.
    args = parser.parse_args(sys.argv[1:])
    video = video_generator(Path(args.video_pathname_str))
    cascade = cv2.CascadeClassifier(args.cascade_pathname_str)
    for frame_filename_str, frame in video:
        rects = cascade.detectMultiScale(frame, 1.2, 3, 0, (20, 20))
        new_rects = []
        for x, y, width, height in rects:
            if not args.noshow:
                frame = cv2.rectangle(
                        frame, (x, y), (x + width, y + height), (0, 255, 0))
            new_rect = Rect(1 * x, 1 * y, 1 * width, 1 * height)
            new_rects.append(new_rect)
        rects_str = ' '.join([
                ','.join(map(str, rect))
                for rect in new_rects])
        sys.stdout.write('{} {}\n'.format(frame_filename_str, rects_str))
        if not args.noshow:
            cv2.imshow('fight!', frame)
            cv2.waitKey(0)


if __name__ == '__main__':
    main()
