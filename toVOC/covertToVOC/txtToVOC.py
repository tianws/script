import sys
sys.path
sys.path.append('../python')

import logging
# import xml.etree.ElementTree as ET

from lxml import etree as ET

import cv2

if sys.version_info.major == 2:
    from pathlib2 import Path
else:
    from pathlib import Path
    unicode = str
assert Path

srcPath = Path('myData/input')
outputPath = Path('myData/output')

srcImgPath = srcPath / 'images'
srcAnoPath = srcPath / 'annotation'
outImgPath = outputPath / 'images'
outAnoPath = outputPath / 'annotation'

show_Label = True

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

def writeXML(frame_name, class_name, rect):
    VOC_label_files = frame_name + '.xml'
    VOC_label_files_path = outAnoPath / VOC_label_files
    pic_name = frame_name + '.jpg'
    pic_file_path = srcImgPath / pic_name
    assert pic_file_path.exists()
    frame = cv2.imread(str(pic_file_path))
    if show_Label:
        label_dir_path = outputPath / 'labeled_img'
        if not label_dir_path.exists():
            label_dir_path.mkdir(parents=True)
        labeled_img_name = 'labeled_' + pic_name
        labeled_img_path = label_dir_path / labeled_img_name
        if labeled_img_path.exists():
            frame = cv2.imread(str(labeled_img_path))
        cv2.rectangle(frame,(rect.x, rect.y),(rect.x + rect.width, rect.y + rect.height),(0, 255, 255))
        cv2.putText(frame,class_name,(rect.x,rect.y), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0,255,255), thickness=2 )
        # cv2.imshow(class_name, frame)
        # cv2.waitKey(1)
        cv2.imwrite(str(labeled_img_path), frame)

    frame_height, frame_width ,frame_depth = frame.shape

    if not VOC_label_files_path.exists():
        xml_root = ET.Element("annotation")
        xml_filename = ET.SubElement(xml_root,"filename")
        xml_filename.text = pic_name
        xml_size = ET.SubElement(xml_root, "size")
        size_width = ET.SubElement(xml_size, "width")
        size_width.text = str(frame_width)
        size_height = ET.SubElement(xml_size, "height")
        size_height.text = str(frame_height)
        size_depth = ET.SubElement(xml_size, "depth")
        size_depth.text = str(frame_depth)
        xml_object = ET.SubElement(xml_root, "object")
        object_name = ET.SubElement(xml_object, "name")
        object_name.text = class_name
        object_pose = ET.SubElement(xml_object, "pose")
        object_pose.text = "Unspecified"
        object_truncated = ET.SubElement(xml_object, "truncated")
        object_truncated.text = "Unspecified"
        object_difficult = ET.SubElement(xml_object, "difficult")
        object_difficult.text = "0"
        object_bndbox = ET.SubElement(xml_object, "bndbox")
        bndbox_xmin = ET.SubElement(object_bndbox, "xmin")
        bndbox_xmin.text = str(rect.x)
        bndbox_ymin = ET.SubElement(object_bndbox, "ymin")
        bndbox_ymin.text = str(rect.y)
        bndbox_xmax = ET.SubElement(object_bndbox, "xmax")
        bndbox_xmax.text = str(rect.x + rect.width)
        bndbox_ymin = ET.SubElement(object_bndbox, "ymin")
        bndbox_ymin.text = str(rect.y + rect.height)

        tree = ET.ElementTree(xml_root)
        tree.write(str(VOC_label_files_path),pretty_print=True)
        logging.info('{} created done...'.format(VOC_label_files_path))
    else:
        parser = ET.XMLParser(remove_blank_text=True)
        updateTree = ET.parse(str(VOC_label_files_path),parser)
        xml_root = updateTree.getroot()
        new_object = ET.Element("object")
        object_name = ET.SubElement(new_object, "name")
        object_name.text = class_name
        object_pose = ET.SubElement(new_object, "pose")
        object_pose.text = "Unspecified"
        object_truncated = ET.SubElement(new_object, "truncated")
        object_truncated.text = "Unspecified"
        object_difficult = ET.SubElement(new_object, "difficult")
        object_difficult.text = "0"
        object_bndbox = ET.SubElement(new_object, "bndbox")
        bndbox_xmin = ET.SubElement(object_bndbox, "xmin")
        bndbox_xmin.text = str(rect.x)
        bndbox_ymin = ET.SubElement(object_bndbox, "ymin")
        bndbox_ymin.text = str(rect.y)
        bndbox_xmax = ET.SubElement(object_bndbox, "xmax")
        bndbox_xmax.text = str(rect.x + rect.width)
        bndbox_ymin = ET.SubElement(object_bndbox, "ymin")
        bndbox_ymin.text = str(rect.y + rect.height)
        xml_root.append(new_object)
        updateTree.write(str(VOC_label_files_path),pretty_print=True)
        logging.info('{} updated done...'.format(VOC_label_files_path))


def convert_xywh_to_2p_label(label):
    x0, y0, w, h = label
    return [x0, y0, x0 + w, y0 + h]

def main():
    assert srcImgPath.exists(), 'srcImgPath path does not exist: {}'.format(str(srcImgPath))
    assert srcAnoPath.exists(), 'srcAnoPath path does not exist: {}'.format(str(srcAnoPath))

    if not outImgPath.exists():
        outImgPath.mkdir(parents=True)
    if not outAnoPath.exists():
        outAnoPath.mkdir(parents=True)

    logging.info('dir checked done ...')

    for one_class_label_file in list(srcAnoPath.glob('*.txt')):
        label_classname = one_class_label_file.stem
        label_dict = {}
        for line in one_class_label_file.open():
            frame_filename_str, rects_str = line.rstrip('\n').split(' ', 1)
            frame_filename_str = frame_filename_str.split('.')[0]
            rects = [Rect(rect_str) for rect_str in rects_str.split(' ')]
            label_dict[frame_filename_str] = rects
            for rect in rects:
                writeXML(frame_filename_str, label_classname, rect)

        logging.info('handle {}\'s label done'.format(label_classname))





if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    main()