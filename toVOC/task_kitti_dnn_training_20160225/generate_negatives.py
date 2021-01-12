import os
import sys
from random import randint
import cv2

sys.path
sys.path.append('../common')


def list_pictures(dir):
    pics = list(os.listdir(dir))
    pics.sort()
    return pics

interest_categories = ["Car", "Truck", "Van", "Tram"]


def convert_2p_to_xywh_label(label):
    x0, y0, x1, y1 = label
    return [x0, y0, x1 - x0, y1 - y0]


def take_labels(label_file, window_size):
    xs, ys = window_size
    interest_label_lines = [line.split(" ") for line in open(
        label_file) if line.split(" ")[0] in interest_categories]
    labels = [map(convert_position_string_to_int, label[4:8])
              for label in interest_label_lines]
    # Label is fucking x0, y0, x1, y1 format.
    return [[label[0] + xs / 2, label[1] + ys / 2, label[2], label[3]]
            for label in map(convert_2p_to_xywh_label, labels)]


def random_rects(max_width, max_height):
    x = randint(0, max_width)
    y = randint(0, max_height)
    width = randint(0, max_width - x)
    height = randint(0, max_height - y)
    return [x, y, width, height]


def convert_position_string_to_int(string):
    return int(float(string))


def fall_into_rec(point, rec):
    px, py = point
    x, y, w, h = rec
    return px >= x and px <= x + w and py >= y and py <= y + h


def is_pos(center, labels):
    for label in labels:
        if fall_into_rec(center, label):
            return True
    return False


def extract_negs(image, size, labels, step=10):
    w, h = size
    img_rows, img_cols = image.shape[:2]
    crops = []
    for x in range(0, img_cols - w, step):
        for y in range(0, img_rows - h, step):
            # sample = image[y:y + h, x:x + w]
            center = (x + w / 2, y + h / 2)
            if is_pos(center, labels):
                pass
                # crops.append([x, y, w, h])
                # is_p.append(True)
                # cv2.circle(image, center, 3, (0, 0, 255))
            else:
                crops.append([x, y, w, h])
                # is_p.append(False)
    return crops


def save_samples(image, sample_rects, neg_dir, pic):
    for idx, crop in enumerate(sample_rects):
        x, y, w, h = crop
        cv2.imwrite(neg_dir + pic + str(idx) +
                    ".jpg", image[y:y + h, x:x + w])


def extract_poss(image, size, labels, step=2):
    crops = []
    sw, sh = size
    for label in labels:
        lx, ly, lw, lh = label
        for center_x in range(int(lx + lw * 0.25), int(lx + lw * 0.75), step):
            for center_y in range(int(ly + lh * 0.25), int(ly + lh * 0.75), step):
                x = center_x - sw / 2
                y = center_y - sh / 2
                w = sw
                h = sh
                crops.append([x, y, w, h])
    return crops


def generate(dir, pos_dir, neg_dir):
    pics_dir = dir + "/image_2/"
    label_dir = dir + "/label_2/"
    pics = list_pictures(pics_dir)
    for idx, pic in enumerate(pics):
        label_file = (label_dir + pic).replace("png", "txt")
        labels = take_labels(label_file, (100, 100))
        image = cv2.copyMakeBorder(cv2.imread(pics_dir + pic), 50, 50,
                                   50, 50, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        negs = extract_negs(image, (100, 100), labels)
        save_samples(image, negs,  neg_dir, pic)
        poss = extract_poss(image, (100, 100), labels)
        save_samples(image, poss, pos_dir, pic)
        # cv2.imshow("ff", image)
        if len(labels) > 0:
            print labels
            print idx, pic
            # exit()

# print list_pictures("/home/qin/Desktop/matlab_kitti/training/image_2/")[0]
import sys
generate(sys.argv[1], sys.argv[2], sys.argv[3])
