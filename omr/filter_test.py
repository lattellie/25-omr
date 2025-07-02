import numpy as np
import cv2
import layers
from pathlib import Path
from typing import Tuple, List
import os
from logger import get_logger
logger = get_logger(__name__)

def loadModel1(img_path:str,model1Data:dict) -> str:
    img_path = Path(img_path)
    f_name = os.path.splitext(img_path.name)[0]
    stems_rests = model1Data['stems_rests']
    clefs_keys = model1Data['clefs_keys']
    notehead = model1Data['notehead']
    symbols = model1Data['symbols']
    staff = model1Data['staff']
    image = model1Data['image']

    layers.register_layer("stems_rests_pred", stems_rests)
    layers.register_layer("clefs_keys_pred", clefs_keys)
    layers.register_layer("notehead_pred", notehead)
    layers.register_layer("symbols_pred", symbols)
    layers.register_layer("staff_pred", staff)
    layers.register_layer("original_image", image)
    logger.info('finish loading all model1\'s data')
    return image, staff, symbols, stems_rests, notehead, clefs_keys

def runModel2(npy_path: str, img_path: str, output_dir: str,img_name = ''):
    # output_dir will be images/tch/tch
    dataDict = np.load(npy_path,allow_pickle=True)
    dataDict = dataDict.tolist()
    image, staff, symbols, stems_rests, notehead, clefs_keys = loadModel1(img_path, dataDict)


    staff_symbols:np.ndarray = staff+symbols
    staff_symbols[staff_symbols>0] = 1
    img = np.ones((staff_symbols.shape[0],staff_symbols.shape[1],3),np.uint16)*255
    xs, ys = np.where(staff_symbols>0)
    img[xs,ys] = (0,0,0)
    # cv2.imwrite(f'{img_name}b&w.jpg',img)
    # b&w is the black and white of the staffs+symbols


    _, img128 = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    # cv2.imwrite(f'{img_name}img128.jpg', img128)
    i_gray = cv2.cvtColor(img128, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5), np.uint8)
    i_dilation_gray = cv2.dilate(i_gray, kernel, iterations = 1)
    # cv2.imwrite(f'{img_name}i_dilation_gray1.jpg',i_dilation_gray)
    notehead_clefkey = notehead+clefs_keys
    xs, ys = np.where(notehead_clefkey>0)
    i_dilation_gray[xs,ys] = 255
    # cv2.imwrite(f'{img_name}i_dilation_gray2.jpg',i_dilation_gray)
    i_dilation_add = i_dilation_gray.copy()
    xs, ys = np.where(stems_rests>0)
    i_dilation_add[xs,ys] = 0
    # cv2.imwrite(f'{img_name}i_dilation_add.jpg',i_dilation_add)
    i_dilation_add_erode = cv2.erode(i_dilation_add, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
    # cv2.imwrite(f'{img_name}i_dilation_add_erode.jpg', i_dilation_add_erode)
    idae_inv = np.invert(i_dilation_add_erode)
    contours, hierachy = cv2.findContours(idae_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    w_thresh = 12
    h_thresh = 6
    idae_bgr = cv2.cvtColor(i_dilation_add_erode, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if w>w_thresh and h>h_thresh:
            idae_bgr = cv2.rectangle(idae_bgr, (x,y), (x+w, y+h), (0,255,0),2)
    cv2.imwrite(f'{img_name}contour{w_thresh}x{h_thresh}.jpg', idae_bgr)
    print('finish dilation')


    
if __name__== '__main__':
    imgname = 'tch'
    # img_path = f'C:/Ellie/ellie2023~2024/iis/omr-iis/images/{imgname}/{imgname}.png'
    base_path = f'C:/Ellie/ellie2023~2024/iis/omr-iis/images/{imgname}/{imgname}'
    runModel2(base_path+'.npy',base_path+'.png',base_path, img_name = f'filter_test/{imgname}_')
