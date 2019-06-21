#coding=utf-8

import cv2
import numpy  as np
from collections import namedtuple
from PIL import Image, ImageDraw, ImageFont

'''Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  '道路(road)'                 ,  7 ,        0 , '道路'            , 1       , False        , False        , (128, 64,128) ),
    Label(  '人行道(sidewalk)'             ,  8 ,        1 , '人行道'            , 1       , False        , False        , (244, 35,232) ),
    Label(  '建筑(building)'             , 11 ,        2 , '建筑'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  '墙(wall)'                 , 12 ,        3 , '墙'    , 2       , False        , False        , (102,102,156) ),
    Label(  '栅栏(fence)'                , 13 ,        4 , '栅栏'    , 2       , False        , False        , (190,153,153) ),
    Label(  '杆(pole)'                 , 17 ,        5 , '杆'          , 3       , False        , False        , (153,153,153) ),
    Label(  '交通灯(traffic light)'        , 19 ,        6 , '交通灯'          , 3       , False        , False        , (250,170, 30) ),
    Label(  '标识牌(traffic sign)'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  '植物(vegetation)'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  '地形(terrain)'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  '天空(sky)'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  '人(person)'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  '骑手(rider)'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  '汽车(car)'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  '卡车(truck)'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  '公交车(bus)'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  '火车(train)'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  '摩托车(motorcycle)'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  '自行车(bicycle)'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
]'''
Label = namedtuple( 'Label' , ['name','trainId', 'color',] )


labels = [
	#	trainId		color
	Label('道路(road)', 0,	(128,  64, 128)),
	Label('杆(pole)',  1,	(244,  35, 232)),
	Label('植物(vegetation)',  2,	(107, 142,  35)),
	Label('井盖(manhole)', 3,	(  0,   0, 200)),
	Label('其他(other)', 4,	(255, 255, 255)),
]



img = np.zeros((1000,1500,3),np.uint8)+255
for l in labels:
	i = l.trainId
	cat = l.name
	color = l.color
	tid = l.trainId
	font = ImageFont.truetype('NotoSansCJK-Black.ttc', 80)
	fillColor = (0,0,0)
	for j in range(3):
		img[i*200+50:i*200+150,50:500,j] = color[2-j]
		img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		position = (600,i*200+30)
		draw = ImageDraw.Draw(img_PIL)    
		draw.text(position, str(tid)+'    '+cat, font=font, fill=fillColor)
		img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
		#cv2.putText(img,cat,(650,i*200+150),cv2.FONT_HERSHEY_COMPLEX,4,(0,0,0),18)
cv2.imwrite('color2calss.png',img)
		

