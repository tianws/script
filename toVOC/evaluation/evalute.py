#!/usr/bin/env python
# coding=utf-8
from __future__ import division, unicode_literals

import argparse
import logging
import phodopus
import squirrel
import sys

if sys.version_info.major == 2:
    from pathlib2 import Path
else:
    from pathlib import Path

if __name__ == '__main__':
    logging.basicConfig(
            format='%(levelname)s:%(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
            description='通过比较现成的识别输出文件（log）和标注文件（txt），以测评车尾或车道的识别效果')
    parser.add_argument('log_pathname', action='store', type=Path)
    parser.add_argument('label_pathname', action='store', type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
            '-s', action='store_const', dest='module',
            const=squirrel, help='squirrel')
    group.add_argument(
            '-p', action='store_const', dest='module',
            const=phodopus, help='phodopus')
    args = parser.parse_args(sys.argv[1:])
    if args.module is None:
        logging.error('You should choose -s or -p! See -h')
    tp_count, fp_count, fn_count = args.module.parse(
            args.log_pathname,
            args.label_pathname)
    precision, recall, fb_measure, _ = args.module.statistics(
            tp_count,
            fp_count,
            fn_count)
    logging.info('precision: {:.3}'.format(precision))
    logging.info('recall: {:.3}'.format(recall))
    logging.info('fb_measure: {:.3}'.format(fb_measure))
