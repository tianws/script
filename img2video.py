import cv2
import glob
import numpy  as np
from collections import namedtuple
from PIL import Image, ImageDraw, ImageFont

# 根据自己的实际情况更改目录。
# 要转换的图片的保存地址，按顺序排好，后面会一张一张按顺序读取。
# convert_image_path = '/home/tianws/Pictures/自动驾驶测试图片/20190517公司附近/autoImages_lane_result/color'

# 帧率(fps)，尺寸(size)，此处设置的fps为24，size为图片的大小，本文转换的图片大小为400×1080，
# 即宽为400，高为1080，要根据自己的情况修改图片大小。
fps = 10
size = (1280, 720)

videoWriter = cv2.VideoWriter('/home/tianws/Pictures/ChangshaTestVideo.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'),
                              fps, size)

with open('changsha_pic_list.txt', 'rt') as f:
    for line in f:
        line = line.strip()
        print(line)
        read_img = cv2.imread(line)
        read_img = cv2.resize(read_img, (1280, 720))
        img_PIL = Image.fromarray(cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB))
        position1 = (20, 20)
        position2 = (20, 120)
        # fillColor = (250, 170, 30)  # 日本
        fillColor = (220, 220, 0)  # 长沙
        # fillColor = (70, 130, 180)  # 北京
        # fillColor = (128, 64, 128)
        font1 = ImageFont.truetype('NotoSansCJK-Black.ttc', 80)
        font2 = ImageFont.truetype('NotoSansCJK-Black.ttc', 70)
        draw = ImageDraw.Draw(img_PIL)
        draw.text(position1, '长沙', font=font1, fill=fillColor)
        draw.text(position2, '双目设备', font=font2, fill=fillColor)
        read_img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', read_img)
        cv2.waitKey(1)
        videoWriter.write(read_img)
videoWriter.release()
