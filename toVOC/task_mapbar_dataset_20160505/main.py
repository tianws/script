import sys
import cv2
from random import shuffle


def parse_line(line):
    segs = line.split(" ")
    pic_name = segs[0]
    rects = []
    for seg in segs[1:]:
        rects.append(map(int, seg.split(",")))
    return pic_name, rects


def load_labels(label_file):
    ret = {}
    for line in open(label_file):
        pic, rects = parse_line(line)
        ret[pic] = rects
    return ret


def extract_poss(size, rects, step=10):
    crops = []
    sw, sh = size
    for rect in rects:
        lx, ly, lw, lh = rect
        for center_x in range(int(lx), int(lx + lw), step):
            for center_y in range(int(ly), int(ly + lh), step):
                x = center_x - sw / 2
                y = center_y - sh / 2
                w = sw
                h = sh
                crops.append([x, y, w, h])
    return crops


def fall_into_rec(point, rec):
    px, py = point
    x, y, w, h = rec
    return px >= x and px <= x + w and py >= y and py <= y + h


def is_pos(center, labels):
    for label in labels:
        if fall_into_rec(center, label):
            return True
    return False


def extract_negs(image, size, rects, step=40):
    w, h = size
    img_rows, img_cols = image.shape[:2]
    crops = []
    for x in range(0, img_cols - w, step):
        for y in range(0, img_rows - h, step):
            # sample = image[y:y + h, x:x + w]
            center = (x + w / 2, y + h / 2)
            if is_pos(center, rects):
                pass
            else:
                crops.append([x, y, w, h])
                # is_p.append(False)
    return crops


def save_samples(image, sample_rects, _dir, pic, offset=(50, 50)):
    xof, yof = offset
    for idx, crop in enumerate(sample_rects):
        x, y, w, h = crop
        x += xof
        y += yof
        cv2.imwrite(_dir + pic + str(idx) +
                    ".jpg", image[y:y + h, x:x + w])


def main(label_file, pics_dir, output_dir):
    labels = load_labels(label_file)
    keys = labels.keys()
    shuffle(keys)
    for idx, pic in enumerate(keys):
        print idx
        rects = labels[pic]
        gray = cv2.cvtColor(cv2.imread(
            pics_dir + "/" + pic), cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 50, 50, 50, 50,
                                  cv2.BORDER_CONSTANT, value=(0))
        print rects
        pos_rects = extract_poss((100, 100), rects)
        cv2.imshow("kk", gray)
        cv2.waitKey(10)
        save_samples(gray, pos_rects, output_dir + "/pos/", pic)
        neg_rects = extract_negs(gray, (100, 100), rects)
        save_samples(gray, neg_rects, output_dir + "/neg/", pic)
        if idx > 4000:
            break


if __name__ == '__main__':
    # Args: Label, pics dir, output dir
    assert len(sys.argv) == 4, "Args: Label, pics dir, output dir"
    main(sys.argv[1], sys.argv[2], sys.argv[3])
