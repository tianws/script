#!/usr/bin/env python
# coding=utf-8
from __future__ import division, unicode_literals

from common import caffe_directory
from common import Path
import logging
import subprocess


def main():
    command_pathname = caffe_directory / 'build/tools/caffe'
    solver_pathname = Path('')
    command = [
            command_pathname,
            'train',
            '-gpu', 'all',
            '-solver', solver_pathname,
            '-shuffle',
    ]
    command = list(map(str, command))
    logging.info(' '.join(command))
    subprocess.call(command)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    main()
