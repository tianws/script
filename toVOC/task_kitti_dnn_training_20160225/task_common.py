import os
interest_categories = ["Car", "Truck", "Van", "Tram"]


def convert_position_string_to_int(string):
    return int(float(string))


def list_pictures(dir):
    pics = list(os.listdir(dir))
    pics.sort()
    return pics


def parse_labels(label_file, interest_categories):
    interest_label_lines = [line.split(" ") for line in open(
        label_file) if line.split(" ")[0] in interest_categories]
    labels = [map(convert_position_string_to_int, label[4:8])
              for label in interest_label_lines]
    return map(convert_2p_to_xywh_label, labels)


def padding_labels(label_file, padding_labels_size):
    pass


def take_labels(label_file, window_size):
    xs, ys = window_size
    interest_label_lines = [line.split(" ") for line in open(
        label_file) if line.split(" ")[0] in interest_categories]
    labels = [map(convert_position_string_to_int, label[4:8])
              for label in interest_label_lines]
    # Label is fucking x0, y0, x1, y1 format.
    return [[label[0] + xs / 2, label[1] + ys / 2, label[2], label[3]]
            for label in map(convert_2p_to_xywh_label, labels)]


def convert_2p_to_xywh_label(label):
    x0, y0, x1, y1 = label
    return [x0, y0, x1 - x0, y1 - y0]
