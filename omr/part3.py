from omr.part2 import OmrStaff
from omr.img_tromr import predict_best
from typing import List
import pickle
import cv2
import numpy as np

def get_bounded_boxes(staffList: List[OmrStaff], image, saveImg=False, output_dir = '', colorList = [(0,0,255)]):
    all_bbox = []
    i = 0
    buffer_length = staffList[i].get_yOne()*2
    bbox = [max(0, staffList[i].get_bbox('l')-int(buffer_length)),
            max(0, staffList[i].get_bbox('t')-int(buffer_length)),
            min(image.shape[1], staffList[0].get_bbox('r')+int(buffer_length)),
            min(staffList[i+1].get_bbox('t'), 
                staffList[i].get_bbox('b')+int(buffer_length))]
    # left, top, right, bottom (x0, y0, x1, y1)
    all_bbox.append(bbox)
    for i in range(1, len(staffList)-1):
        buffer_length = staffList[i].get_yOne()*2
        bbox = [max(0, staffList[i].get_bbox('l')-int(buffer_length)),
                max(staffList[i-1].get_bbox('b'), 
                    staffList[i].get_bbox('t')-int(buffer_length)),
                min(image.shape[1], staffList[i].get_bbox('r')+int(buffer_length)),
                min(staffList[i+1].get_bbox('t'), 
                    staffList[i].get_bbox('b')+int(buffer_length))]
        all_bbox.append(bbox)
    i = i+1
    buffer_length = staffList[i].get_yOne()*2
    bbox = [max(0, staffList[i].get_bbox('l')-int(buffer_length)),
            max(staffList[i-1].get_bbox('b'), 
                staffList[i].get_bbox('t')-int(buffer_length)),
            min(image.shape[1], staffList[i].get_bbox('r')+int(buffer_length)),
            min(image.shape[0], staffList[i].get_bbox('b')+int(buffer_length))]
    all_bbox.append(bbox)
    if saveImg:
        img = image.copy()
        for i in range(len(all_bbox)):
            b = all_bbox[i]
            img = cv2.rectangle(img, (b[0],b[1]), (b[2],b[3]), colorList[i%len(colorList)],2)
        cv2.imwrite(output_dir+'_modifiedRect.jpg', img)

    return all_bbox

def runModel3(pkl_path, img_path, output_dir, saveCrop = False):
    image = cv2.imread(img_path)
    with open(pkl_path, 'rb') as file:
        staffList = pickle.load(file)
    img = image.copy()
    colorList = [(0,0,255),(255,255,0),(255,0,255), (0,255,0)]
    for i in range(len(staffList)):
        staffList[i].expand_bound()
        img = staffList[i].drawRect(img, output_dir, saveImg = True, col = colorList[i%len(colorList)])
    
    assert len(staffList) >= 3
    all_bbox = get_bounded_boxes(staffList,image,saveImg=True, output_dir=output_dir, colorList = colorList)
    assert len(all_bbox) == len(staffList)

    predict_list = []
    for i in range(len(staffList)):
        b = all_bbox[i]
        img = image[b[1]:b[3], b[0]:b[2],:]
        if saveCrop:
            cv2.imwrite(output_dir+f'_crop{i}.jpg',img)
        outstr = predict_best(img, img_path, saveTxt=False)
        predict_list.append(outstr)
    print(predict_list)
    # save to .npy format
    np.save(output_dir+'_text.npy', predict_list)
    # save to readable format
    with open(output_dir+'.txt', 'w') as file:
        for predict_str in predict_list:
            file.write(predict_str+"\n\n")
    print(output_dir)
    return predict_list