import cv2
import numpy as np
def saveBinaryImage(ndarray, image, name, color = (255,0,0)):
    img = image.copy()
    img = img//10+225
    xs, ys = np.where(ndarray>0)
    img[xs, ys] = color
    cv2.imwrite(name+'.jpg', img)