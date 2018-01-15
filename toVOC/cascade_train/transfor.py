#!/usr/bin/env python2
# coding=utf-8

import argparse
import shutil
import sys
import tempfile


if sys.version_info.major == 2:
    from pathlib2 import Path
else:
    from pathlib import Path


def convert(rect_str):
    """把一条字符串转换为二级列表
    rect_str: str, '561 281 33 25 608 285 32 26'
    return: [[561, 201, 33, 25], [608, 285, 32, 26]]
    """
    rect_num_list = [int(string) for string in rect_str.split()]
    return [rect_num_list[index:index + 4]
            for index in range(0, len(rect_num_list), 4)]


def main():
    parser = argparse.ArgumentParser(
            description='转换标准标注为 cascade_createsamples 所用的格式')
    parser.add_argument('label_pathname', action='store', type=Path)
    args = parser.parse_args(sys.argv[1:])
    with tempfile.NamedTemporaryFile('r+') as tmp_file, \
            args.label_pathname.open as label_file:
        for line in label_file:
            filename, _, rect_str = line.split(' ', 2)
            rects = convert(rect_str)
            new_rect_str = ' '.join(
                    [','.join(map(str, rect)) for rect in rects])
            new_line = '{} {}\n'.format(filename, new_rect_str)
            tmp_file.write(new_line)
        tmp_file.seek(0)
        shutil.copy(tmp_file.name, '{}/new_{}'.format(
                args.label_pathname.parent, args.label_pathname.name))


if __name__ == '__main__':
    main()
