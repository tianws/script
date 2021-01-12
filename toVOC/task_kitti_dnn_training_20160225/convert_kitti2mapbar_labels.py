import os
import sys
from random import randint
import cv2
from task_common import *

sys.path
sys.path.append('../common')


def main(input_dir):
    pics_dir = input_dir + "/image_2/"
    label_dir = input_dir + "/label_2/"
    pics = list_pictures(pics_dir)
    for idx, pic in enumerate(pics):
        label_file = (label_dir + pic).replace("png", "txt")
        labels = parse_labels(label_file, interest_categories)
        print pic + " " + \
            " ".join([",".join(map(str, label)) for label in labels])
        # of.write(str)

if __name__ == '__main__':
    main(sys.argv[1])
