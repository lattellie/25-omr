import pickle
import numpy as np
import enum
import cv2
from omr.part2 import OmrStaff
from typing import List

class ClefType(enum.Enum):
    G_CLEF = 1
    F_CLEF = 2

class SfnType(enum.Enum):
    FLAT = 1
    SHARP = 2
    NATURAL = 3

class RestType(enum.Enum):
    WHOLE_HALF = 1
    QUARTER = 2
    EIGHTH = 3
    SIXTEENTH = 4
    THIRTY_SECOND = 5
    SIXTY_FOURTH = 6
    WHOLE = 7
    HALF = 8

# map each string to OmrStaff object
def map_staff_pred(staff:OmrStaff, pred_str:str):
    str_lst = pred_str.split('+')
    # now str_lst is a list of ['clef-G2'] etc
    # staff is a OmrStaff Object
    elements: List[OmrStaff.StaffElements] = staff.elements
    # elements: a list of OmrStaff.StaffElements
    # we are iterating through the str_lst
    noteValDict = {'C':-1,'D':0,'E':1,'F':2,'G':3,'A':4,'B':5}

    currlst_idx = 0
    startbox_idx = 0
    unclassified_list_idx = []
    for i in range(len(str_lst)):
        currStr = str_lst[i]
        currbox_idx = startbox_idx
        # barline
        if currStr.startswith('b'):
            while(currbox_idx<len(elements)):
                if (elements[currbox_idx].typ != "barline"):
                    currbox_idx += 1
                else:
                    break
            if currbox_idx>=len(elements):
                break
            if elements[currbox_idx].typ == 'barline':
                unclassified_list_idx = unclassified_list_idx+[idx for idx in range(startbox_idx, currbox_idx)]
                elements[currbox_idx].addLabel(currStr)
                startbox_idx = currbox_idx+1
            continue
        # clef
        if currStr.startswith('c'):
            while (currbox_idx<len(elements)):
                if (elements[currbox_idx].typ != 'clef') & (elements[currbox_idx].typ != "barline"):
                    currbox_idx += 1
                else:
                    break
            if elements[currbox_idx].typ == 'clef':
                unclassified_list_idx = unclassified_list_idx+[idx for idx in range(startbox_idx, currbox_idx)]
                elements[currbox_idx].addLabel(currStr)
                startbox_idx = currbox_idx+1
            continue
        # key signature: assign everything to it until we find a note/rest/barline
        elif currStr.startswith('k'):
            while(currbox_idx<len(elements)):
                if (elements[currbox_idx].typ != 'note') & (elements[currbox_idx].typ != "barline"):
                    elements[currbox_idx].addLabel(currStr)
                    currbox_idx += 1
                else:
                    break
            # currbox_idx is note or rest
            startbox_idx = currbox_idx
            continue
        # rest
        elif currStr.startswith('r'):
            while(currbox_idx<len(elements)):
                if (elements[currbox_idx].typ != 'rest') & (elements[currbox_idx].typ != "barline"):
                    currbox_idx += 1
                else:
                    break
            if elements[currbox_idx].typ == 'rest':
                unclassified_list_idx = unclassified_list_idx+[idx for idx in range(startbox_idx, currbox_idx)]
                elements[currbox_idx].addLabel(currStr)
                startbox_idx = currbox_idx+1
            continue
        elif currStr.startswith('n'):
            oriCurrStr = currStr
            maxIdx=currbox_idx
            for currStr in oriCurrStr.split('|'):
                currbox_idx = startbox_idx
                foundNote = False
                while (not foundNote) & (currbox_idx<len(elements)):
                    if elements[currbox_idx].typ == "barline":
                        break
                    while(currbox_idx<len(elements)):
                        if (elements[currbox_idx].typ != 'note') & (elements[currbox_idx].typ != "barline"):
                            currbox_idx += 1
                        else:
                            break
                    if currbox_idx>=len(elements):
                        break
                    if elements[currbox_idx].typ == 'note':
                        try:
                            noteval = noteValDict[currStr[5]]+(int(currStr[6])-4)*7
                            if noteval-elements[currbox_idx].addval == 0:
                                unclassified_list_idx = unclassified_list_idx+[idx for idx in range(startbox_idx, currbox_idx)]
                                elements[currbox_idx].addLabel(currStr)
                                maxIdx = max(maxIdx,currbox_idx)
                                foundNote = True
                            else:
                                currbox_idx += 1
                        except:
                            print(f"{currStr} doesn't match the value evaluation of {currStr[5]} and {currStr[6]}")
                            pass
                    continue
                continue
            startbox_idx = maxIdx+1
        else:
            continue

def save_pred(elements: List[OmrStaff.StaffElements], image, output_path = ''):
    img = image.copy()
    for elm in elements:
        if elm.newLabel is not None:
            if elm.typ == 'note':
                if 'eighth' in elm.newLabel:
                    color = (0,0,255)
                elif 'sixteenth' in elm.newLabel:
                    color = (255,255,0)
                elif 'quarter' in elm.newLabel:
                    color = (255,0,120)
                else:
                    print(elm.newLabel)
                    color = (0,150,0)
                img = cv2.rectangle(img, (elm.bbox[0], elm.bbox[1]), (elm.bbox[2],elm.bbox[3]), color, 2)
                img = cv2.putText(img, elm.newLabel[5:8], (elm.bbox[0], elm.bbox[1]-5) , cv2.FONT_HERSHEY_SIMPLEX,  0.7, color, 2, cv2.LINE_AA)
            else: 
                img = cv2.rectangle(img, (elm.bbox[0], elm.bbox[1]), (elm.bbox[2],elm.bbox[3]), (255,0,0), 1)
                img = cv2.putText(img, elm.newLabel[0:3], (elm.bbox[0], elm.bbox[1]-5) , cv2.FONT_HERSHEY_SIMPLEX,  0.7, (255,0,0), 2, cv2.LINE_AA)
    if output_path.endswith('.jpg'):
        cv2.imwrite(output_path, img)
    return img

def runModel4(npy_path, pkl_path, img_path, output_dir):
    with open(pkl_path, 'rb') as file:
        staffList = pickle.load(file)
    pred_list = np.load(npy_path,allow_pickle=True)

    image = cv2.imread(img_path)
    img = image.copy()
    for i in [4]:
        staff = staffList[i]
        pred_str = str(pred_list[i])
        elements = staff.elements
        # staff.print_staff()
        map_staff_pred(staff,pred_str)
        # staff.print_staff()
        save_pred(elements, image, output_path=str(i)+'.jpg')
        img = save_pred(elements, img, output_path='allInOne.jpg')
    print(npy_path)
    print(img_path)
    print(output_dir)
    return ''