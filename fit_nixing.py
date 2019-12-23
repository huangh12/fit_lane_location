#coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np
import matplotlib.pyplot as plt
import cPickle
from easydict import EasyDict as edict
from matplotlib.pyplot import MultipleLocator
from bfs_group import bfs_clustering
import cv2
import glob
from random import random as rand
from PIL import Image, ImageDraw, ImageFont
import json
import os

config = edict()
config.minimum_points = 50
config.max_group = 3
config.max_neighbor_distance = 10
config.resize_factor = 0.5

color_map = {'White':'白色', 'Silver_gray': '银灰色', 'Black': '黑色', 'Red': '红色', 'Brown': '棕色', 'Blue': '蓝色',
             'Yellow': '黄色', 'Purple': '紫色', 'Green': '绿色', 'Pink': '粉色', 'Ching': '青色', 'Golden': '金色', 'other': '其他'}

letter = [u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'J', u'K', u'L', u'M',
          u'N', u'P', u'Q', u'R', u'S', u'T', u'U', u'V', u'W', u'X', u'Y', u'Z']
province = [u'京', u'津', u'沪', u'渝', u'黑', u'吉', u'辽', u'冀', u'晋', u'鲁', u'豫', u'陕', u'甘', u'青', u'苏', u'浙',
            u'皖', u'鄂', u'湘', u'闽', u'赣', u'川', u'贵', u'云', u'粤', u'琼', u'蒙', u'宁', u'新', u'桂', u'藏']

type_map = {'BigTruck': '货车', 'Bus': '公交车', 'Lorry': '货车', 'MPV': '轿车', 'MiniVan': '轿车', 'MiniBus': '公交车',
            'SUV': '轿车', 'Scooter': '轿车', 'Sedan_Car': '轿车', 'Special_vehicle': '其他', 'Three_Wheeled_Truck':'其他', 'other': '其他', 'Minibus': '公交车'}


def draw_box_v2(img, box, alphaReserve=0.8, color=None):
    color = (rand() * 255, rand() * 255, rand() * 255) if color is None else color
    h,w,_ = img.shape
    x1 = max(0, int(float(box[0])))
    y1 = max(0, int(float(box[1])))
    x2 = min(w-1, int(float(box[2])))
    y2 = min(h-1, int(float(box[3])))
    B, G, R = color
    img[y1:y2, x1:x2, 0] = img[y1:y2, x1:x2, 0] * alphaReserve + B * (1 - alphaReserve)
    img[y1:y2, x1:x2, 1] = img[y1:y2, x1:x2, 1] * alphaReserve + G * (1 - alphaReserve)
    img[y1:y2, x1:x2, 2] = img[y1:y2, x1:x2, 2] * alphaReserve + R * (1 - alphaReserve)
    cv2.line(img, (x1, y1), (x1+7, y1), (255,255,255), thickness=1)
    cv2.line(img, (x1, y1), (x1, y1+7), (255,255,255), thickness=1)
    cv2.line(img, (x2, y1), (x2-7, y1), (255,255,255), thickness=1)
    cv2.line(img, (x2, y1), (x2, y1+7), (255,255,255), thickness=1)
    cv2.line(img, (x1, y2), (x1+7, y2), (255,255,255), thickness=1)
    cv2.line(img, (x1, y2), (x1, y2-7), (255,255,255), thickness=1)
    cv2.line(img, (x2, y2), (x2-7, y2), (255,255,255), thickness=1)
    cv2.line(img, (x2, y2), (x2, y2-7), (255,255,255), thickness=1)

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20, font_path="./LiHeiPro.ttf"):
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(font_path, textSize, encoding="utf-8")
    draw.text((left, top), unicode(text.decode('utf-8')) , textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def draw_history(blend_img, history, history_cnt, history_record, history_platenum):
    history = [_ for i, _ in enumerate(history) if history_cnt[i]>0]
    history_record = [_ for i, _ in enumerate(history_record) if history_cnt[i]>0]
    history_platenum = [_ for i, _ in enumerate(history_platenum) if history_cnt[i]>0]
    history_cnt = [_-1 for i, _ in enumerate(history_cnt) if history_cnt[i]>0]
    for i, plate in enumerate(history):
        ph, pw = plate.shape[:2]
        if 70+50*i+ph >= blend_img.shape[0]:
            continue
        blend_img[70+50*i:70+50*i+ph,w-290:w-290+pw,:] = plate
        text = '违章记录:第%d帧' %history_record[i]
        blend_img = cv2ImgAddText(blend_img, text, w-290+pw+10,70+50*i+5, textColor=(0, 0, 0),\
                    textSize=20, font_path="./LiHeiPro.ttf")        
        if history_platenum[i] != ' ':
            text = '车牌识别:'+ history_platenum[i]
            blend_img = cv2ImgAddText(blend_img, text, w-290+pw+10,70+50*i+25, textColor=(0, 0, 0),\
                        textSize=20, font_path="./LiHeiPro.ttf")
    return blend_img, history, history_cnt, history_record, history_platenum

def cal_iou(box1, box2):
    iw = min(box1[2], box2[2]) - max(box1[0], box2[0]) + 1
    if iw > 0:
        ih = min(box1[3], box2[3]) - max(box1[1], box2[1]) + 1
        if ih > 0:
            box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
            box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
            all_area = float(box1_area + box2_area - iw * ih)
            return iw * ih / all_area
    return 0

# judge whether line segment (xc,yc)->(xr,yr) is crossed with infinite line (x1,y1)->(x2,y2)
def is_cross(xc,yc,xr,yr,x1,y1,x2,y2):
    if x1 == x2:
        if (xc-x1) * (xr-x1) < 0:
            return True
        else:
            return False
    return ((y2-y1)/(x2-x1)*(xc-x1)+y1-yc) * \
           ((y2-y1)/(x2-x1)*(xr-x1)+y1-yr) < 0

def filter_area(boxes, area=50):
    if len(boxes) > 0:
        return np.where((boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0]) > area**2)[0]
    else:
        return np.array([], dtype=np.int)

def indicator(x):
    x_square_sum, x_sum = np.sum(x**2), np.sum(x)
    det = len(x) * x_square_sum - x_sum**2
    return x_square_sum, x_sum, det

def solve_k_b(x, y):
    x_square_sum, x_sum, det = indicator(x)
    while det == 0:
        x = x[:-1]
        y = y[:-1]
        x_square_sum, x_sum, det = indicator(x)
    N_ = len(x)
    k_ = np.sum(y * (N_*x-x_sum)) / det
    b_ = np.sum(y * (x_square_sum-x*x_sum)) / det
    return N_, k_, b_


if __name__ == "__main__":
    json_path = 'nixing/nixingattrs.json'
    boxes_results = []
    with open(json_path, 'r') as f:
        line = f.readline()
        while line:
            this_img = json.loads(line.strip())
            boxes_results.append(this_img)
            line = f.readline()

    save_dir = 'nixing_v3'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    os.system('rm ./*.jpg ./*.png ./%s/*.jpg' %save_dir)

    with open('nixing/nixing_mask_res.pkl', 'rb') as f:
        img_list = cPickle.load(f)['all_seg_results']
        img_list = [_['seg_results'] for _ in img_list]

    img_dir = './nixing/frames'
    num_img = len(os.listdir(img_dir))

    history = []
    history_cnt = []
    history_record = []
    history_platenum = []
    for cnt in range(num_img):
        print('%d/%d' %(cnt,num_img))
        # if cnt < 110:
            # continue
        img = img_list[cnt]

        im_path = os.path.join(img_dir, 'nixing.mp4_%06d.jpg' %(cnt+1))
        raw_img = cv2.imread(im_path)

        lane_img = 255 * np.ones_like(raw_img, dtype=np.uint8)
        lane_img[np.where(img == 1)] = [0,225,0]
        lane_img[np.where(img == 2)] = [0,225,255]
        
        blend_img = cv2.addWeighted(raw_img, 1.0, lane_img, 0.3, gamma=0)

        # parse the boxes (vehicle box, plate box, vehicle head box, vehicle tail box)
        vehicle_boxes = [_['data'] for _ in boxes_results[cnt]['vehicle']]
        vehicle_attrs = [_['attrs'] for _ in boxes_results[cnt]['vehicle']]
        plate_data = boxes_results[cnt]['plate_box']
        if plate_data != []:
            plate_boxes = [_['data'] for _ in plate_data]
            plate_nums = [_['attrs']['plate_num']]
            for i in range(len(plate_nums)):
                if len(plate_nums[i]) >= 7 and plate_nums[i][0] in province and plate_nums[i][1] in letter:
                    plate_nums.append(plate_nums[i])
                else:
                    plate_nums[i] = ' '
                print(plate_nums[-1])
        else:
            plate_boxes, plate_nums = [], []

        head_box, tail_box = [], []
        for item in boxes_results[cnt]['common_box']:
            if item['attrs']['head'] == 'tail':
                tail_box.append(item['data'])
            elif item['attrs']['head'] == 'head':
                head_box.append(item['data'])
            else:
                raise ValueError('unsupported attr!')

        # draw the boxes (vehicle box, plate box, vehicle head box, vehicle tail box)
        for box, attrs in zip(vehicle_boxes, vehicle_attrs):
            draw_box_v2(blend_img, box, color=(255,0,0), alphaReserve=0.9)
            text = color_map[attrs['color']]
            text += type_map[attrs['type']]
            cv2.rectangle(blend_img, (int(box[0]), int(box[1])-20), (int(box[0])+70, int(box[1])), (128, 128, 128), thickness=-1)
            blend_img = cv2ImgAddText(blend_img, text, int(box[0]), int(box[1]-20), textColor=(255, 255, 255),\
                            textSize=15, font_path="./LiHeiPro.ttf")
        for box in plate_boxes:
            draw_box_v2(blend_img, box, color=(0,0,255), alphaReserve=0.7)
        for box in head_box:
            draw_box_v2(blend_img, box, color=(0,0,128), alphaReserve=0.7)      
        for box in tail_box:
            draw_box_v2(blend_img, box, color=(0,0,128))

        # cluster the lane points
        neighbor = list(range(1, config.max_neighbor_distance+1))
        neighbor.extend([-i for i in neighbor])
        neighbor.append(0)
        dsize = (int(img.shape[1]*config.resize_factor), int(img.shape[0]*config.resize_factor))
        resized_img = cv2.resize(img, dsize, fx=config.resize_factor,fy=config.resize_factor)
        group_res = bfs_clustering(resized_img, neighbor, ig_cls=0, show=False)
        h, w = img.shape[:2]
        resized_h, resized_w = resized_img.shape[:2]

        # title = '基于X2的"去中心化"违章记录仪'
        # blend_img = cv2ImgAddText(blend_img, title, 20,20, textColor=(0, 0, 0),\
        #                 textSize=45, font_path="./LiHeiPro.ttf")

        title = '逆行车辆:'
        blend_img = cv2ImgAddText(blend_img, title, w-200,20, textColor=(255, 0, 0),\
                        textSize=25, font_path="./LiHeiPro.ttf")

        lanes = []
        b = []
        for cls in group_res:
            print('----cls %d----' %cls)
            for g in group_res[cls]:
                if len(g) < config.minimum_points:
                    continue        
                print('group length: %d' %(len(g)))
                x, y = [], []
                for i, j in g:
                    x.append(j)
                    y.append(resized_h-1-i)
                x = np.array(x, dtype='float32') / config.resize_factor
                y = np.array(y, dtype='float32') / config.resize_factor

                N_, k_, b_ = solve_k_b(x, y)
                print(N_, k_, b_)

                x1, x2 = np.min(x), np.max(x)
                y1, y2 = k_ * x1 + b_, k_ * x2 + b_
                y1, y2 = h-1-y1, h-1-y2
                if cls == 1:
                    color = (0,225,0)
                else:
                    color = (0,225,225)
                    if k_ > 0.1:
                        lanes.append([x1,y1,x2,y2])
                        b.append(b_)
                        # cv2.line(blend_img,(int(x1),int(y1)),(int(x2),int(y2)), color, thickness=3)

        # find the central yellow solid line
        lane = lanes[np.argmax(-1 * np.array(b))]

        # judge whether cross solid lane
        for box in head_box:
            if (box[2] - box[0] + 1) * (box[3] - box[1] + 1) < 50*50:
                continue
            ref_line = [0,0,(box[0]+box[2])/2,(box[1]+box[3])/2]  # (x1,y2,x2,y2)
            input1 = ref_line + lane
            if is_cross(*input1):
                text = '逆行危险！'
                print(text)
                blend_img = cv2ImgAddText(blend_img, text, int((box[0]+box[2])/2-20),int(box[1]), textColor=(255, 0, 0),\
                            textSize=15, font_path="./LiHeiPro.ttf")
                ious = np.array([cal_iou(_, box) for _ in plate_boxes])
                if ious.size > 0:
                    max_idx = np.argmax(ious)
                    pbox = plate_boxes[max_idx]
                    pnum = plate_nums[max_idx]
                    pbox[0] -= 10
                    pbox[2] += 10
                    pbox[1] -= 10
                    pbox[3] += 10
                    ratio = (pbox[3]-pbox[1]) / (pbox[2]-pbox[0])
                    ph = 50
                    pw = int(ph / ratio)
                    pbox = [int(_) for _ in pbox]
                    plate = raw_img[pbox[1]:pbox[3],pbox[0]:pbox[2],:]
                    plate = cv2.resize(plate, (pw,ph))
                    history.insert(0, plate)
                    history_cnt.insert(0, 1)
                    history_record.insert(0, cnt)
                    history_platenum.insert(0, pnum)

        blend_img, history, history_cnt, history_record, history_platenum = \
            draw_history(blend_img, history, history_cnt, history_record, history_platenum)

        cv2.imwrite('./%s/tmp%d.jpg' %(save_dir,cnt), blend_img)