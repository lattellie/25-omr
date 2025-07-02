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

from omr.logger import get_logger

logger = get_logger(__name__)

noteValDictG2 = {'C':-1,'D':0,'E':1,'F':2,'G':3,'A':4,'B':5}
noteVectG2 = ['D','E','F','G','A','B','C']

noteValDictF4 = {'C':-3,'D':-2,'E':-1,'F':0,'G':1,'A':2,'B':3}
noteVectF4 = ['F','G','A','B','C','D','E']
# turn note-C5_quarter or rest-quarter to n6, r0
def str_to_imgCode_G2(original_note:str)->str:
    if original_note.startswith('rest'):
        # use 0 for better future debug
        return 'r0'
    elif original_note.startswith('note'):
        return 'n'+str(noteValDictG2[original_note[5]]+(int(original_note[6])-4)*7)
    else:
        assert original_note.startswith('nonote')
def str_to_imgCode_F4(original_note:str)->str:
    if original_note.startswith('rest'):
        # use 0 for better future debug
        return 'r0'
    elif original_note.startswith('note'):
        return 'n'+str(noteValDictF4[original_note[5]]+(int(original_note[6])-2)*7)
    else:
        assert original_note.startswith('nonote')

# turn n6, n12 to C5 D6 etc
def imgCode_to_str_G2(imgCode:str)->str:
    if imgCode.startswith('n'):
        note_disp = noteVectG2[int(imgCode[1:])%len(noteVectG2)]
        note_height = (int(imgCode[1:])+1)//7+4
        return note_disp+str(note_height)
    else:
        assert imgCode.startswith('n')
def imgCode_to_str_F4(imgCode:str)->str:
    if imgCode.startswith('n'):
        note_disp = noteVectF4[int(imgCode[1:])%len(noteVectG2)]
        note_height = (int(imgCode[1:])+1)//7+2
        return note_disp+str(note_height)
    else:
        assert imgCode.startswith('n')

# find the first occurance of a certain note/rest (used for notes with '|')
def find_index(my_list, element):
    try:
        return my_list.index(element)
    except ValueError:
        return -1
    
# list_str: ['clef-G2', 'keySignature-EM'...etc]
# list_img_str: ['c','k','n6','n-1' ... etc]
# list_str_idx: [0,1, ... ], length = list_str's length
# list_img_idx: [0,1, ... ], length = list_img's length
# return a numpy of [[img's idx, str's idx], [img's idx, str's idx]]
def LongestContinuousG2_recur(list_str:List[str], list_img_str:List[str], list_str_idx: List[int], list_img_idx: List[int], solutions)->np.array:
    inputs = frozenset([tuple(list_str), tuple(list_img_str)])
    solved = solutions.get(inputs, None)
    if solved is not None:
        return solved
    
    if len(list_str)==0 or len(list_img_str)==0:
        solved = np.empty((0,3), dtype=int)
        solutions[inputs] = solved
        return solved

    currStr = list_str[0] # ex: note-C5_quarter
    currImg = list_img_str[0] # ex: n6
    assert len(list_img_idx) == len(list_img_str)
    assert len(list_str_idx) == len(list_str)
    currStrTyp = currStr.split('-')[0]

    # note or rest or clef or barline or keySignature or TimeSignature (ignore for now)
    if currStrTyp.startswith('note') or currStrTyp.startswith('rest'):
        # has one note or multiple notes
        if '|' in currStr:
            # order of note1, note2 for note1|note2
            note_lst = [n for n in currStr.split('|') if (n.startswith('n') or n.startswith('r'))]            
            order_lst = [find_index(list_img_str, str_to_imgCode_G2(n)) for n in note_lst]
            sorted_indices = [i[0] for i in sorted(enumerate(order_lst), key=lambda x: x[1])]
            list_str.pop(0) # remove the note1|note2| ... etc
            group_idx = list_str_idx.pop(0) # remove the corresponding idx
            for i in range(len(sorted_indices)):
                list_str.insert(0, note_lst[sorted_indices[len(sorted_indices)-i-1]])
                list_str_idx.insert(0, group_idx)
            solved = LongestContinuousG2_recur(list_str, list_img_str, list_str_idx, list_img_idx, solutions)
        else:
            # n6, r, n8 etc
            noteCode = str_to_imgCode_G2(currStr)
            if noteCode == currImg:
                solved = np.vstack([[list_img_idx[0], list_str_idx[0],0],LongestContinuousG2_recur(list_str[1:], list_img_str[1:],list_str_idx[1:], list_img_idx[1:], solutions)])
            elif (noteCode.startswith('n') and currImg.startswith('s')):
                solved = np.vstack([[list_img_idx[0], list_str_idx[0],0],LongestContinuousG2_recur(list_str, list_img_str[1:],list_str_idx, list_img_idx[1:], solutions)])
            else:
                notenum = int(noteCode[1:])
                opt3 = np.empty((0,3), dtype=int)
                if noteCode.startswith('n') and (notenum>12 or notenum<-2):
                    if (currImg == 'n'+str(notenum+1)) or (currImg == 'n'+str(notenum-1)):
                        opt3 = np.vstack([[list_img_idx[0], list_str_idx[0],1],LongestContinuousG2_recur(list_str[1:], list_img_str[1:],list_str_idx[1:], list_img_idx[1:], solutions)])
                assert len(list_img_idx) == len(list_img_str)
                opt1 = LongestContinuousG2_recur(list_str[1:], list_img_str,list_str_idx[1:], list_img_idx, solutions)
                opt2 = LongestContinuousG2_recur(list_str, list_img_str[1:],list_str_idx, list_img_idx[1:],solutions)
                lst = [len(opt1), len(opt2),len(opt3)]
                maxIdx = [i for i,v in enumerate(lst) if v==max(lst)]
                if len(maxIdx)==1:
                    chosen = maxIdx[0]
                else:
                    # TODO
                    logger.debug(f"there are {len(maxIdx)} options with same length")
                    lstsum = [np.sum(opt1.T[2]), np.sum(opt2.T[2]), np.sum(opt3.T[2])]
                    # filtered_lst is the number of "displacement sum" of those in maxIdx
                    filtered_lst = [lstsum[i] for i in maxIdx]
                    min_filt_idx = [i for i,v in enumerate(filtered_lst) if v==min(filtered_lst)]
                    chosen = maxIdx[min_filt_idx[0]]
                if chosen == 0:
                    solved = opt1
                elif chosen == 1:
                    solved = opt2
                else:
                    solved = opt3
    # key signature
    elif currStrTyp.startswith('k'):
        if currImg.startswith('c') or currImg.startswith('s'):
            # don't remove the str cause it might still be the next one's class
            solved = np.vstack([[list_img_idx[0], list_str_idx[0],0],LongestContinuousG2_recur(list_str, list_img_str[1:],list_str_idx, list_img_idx[1:], solutions)])
        else:
            # remove the str and keep the image?
            opt1 = LongestContinuousG2_recur(list_str[1:], list_img_str,list_str_idx[1:], list_img_idx, solutions)
            opt2 = LongestContinuousG2_recur(list_str, list_img_str[1:],list_str_idx, list_img_idx[1:], solutions)
            if len(opt1)>len(opt2):
                solved = opt1
            elif len(opt1)==len(opt2):
                if np.sum(opt1.T[2]) > np.sum(opt2.T[2]):
                    solved = opt2
                else:
                    solved = opt1
            else:
                solved = opt2
    # clef and barline
    elif currStrTyp.startswith('c') or currStrTyp.startswith('b'):
        if currImg.startswith(currStrTyp[0]):
            solved = np.vstack([[list_img_idx[0], list_str_idx[0],0],LongestContinuousG2_recur(list_str[1:], list_img_str[1:],list_str_idx[1:], list_img_idx[1:], solutions)])
        else:
            opt1 = LongestContinuousG2_recur(list_str[1:], list_img_str,list_str_idx[1:], list_img_idx, solutions)
            opt2 = LongestContinuousG2_recur(list_str, list_img_str[1:],list_str_idx, list_img_idx[1:], solutions)
            if len(opt1)>len(opt2):
                solved = opt1
            elif len(opt1)==len(opt2):
                if np.sum(opt1.T[2]) > np.sum(opt2.T[2]):
                    solved = opt2
                else:
                    solved = opt1
            else:
                solved = opt2
    else:
        logger.debug(currStrTyp)
        # only will have this if it's a time signature, move on for now (skip the time signature)
        # or n for nonote
        # or m for multirest
        assert (currStrTyp.startswith('timeSignature') or currStrTyp.startswith('nonote') or currStrTyp.startswith('multirest'))
        solved = LongestContinuousG2_recur(list_str[1:], list_img_str,list_str_idx[1:], list_img_idx, solutions)
    solutions[inputs] = solved
    return solved
def LongestContinuousF4_recur(list_str:List[str], list_img_str:List[str], list_str_idx: List[int], list_img_idx: List[int], solutions)->np.array:
    
    inputs = frozenset([tuple(list_str), tuple(list_img_str)])
    solved = solutions.get(inputs, None)
    if solved is not None:
        return solved
    
    if len(list_str)==0 or len(list_img_str)==0:
        solved = np.empty((0,3), dtype=int)
        solutions[inputs] = solved
        return solved

    currStr = list_str[0] # ex: note-C5_quarter
    currImg = list_img_str[0] # ex: n6
    assert len(list_img_idx) == len(list_img_str)
    assert len(list_str_idx) == len(list_str)
    currStrTyp = currStr.split('-')[0]

    # note or rest or clef or barline or keySignature or TimeSignature (ignore for now)
    if currStrTyp.startswith('note') or currStrTyp.startswith('rest'):
        # has one note or multiple notes
        if '|' in currStr:
            # order of note1, note2 for note1|note2
            note_lst = [n for n in currStr.split('|') if (n.startswith('n') or n.startswith('r'))]            
            order_lst = [find_index(list_img_str, str_to_imgCode_F4(n)) for n in note_lst]
            sorted_indices = [i[0] for i in sorted(enumerate(order_lst), key=lambda x: x[1])]
            list_str.pop(0) # remove the note1|note2| ... etc
            group_idx = list_str_idx.pop(0) # remove the corresponding idx
            for i in range(len(sorted_indices)):
                list_str.insert(0, note_lst[sorted_indices[len(sorted_indices)-i-1]])
                list_str_idx.insert(0, group_idx)
            solved = LongestContinuousF4_recur(list_str, list_img_str, list_str_idx, list_img_idx, solutions)
        else:
            # n6, r, n8 etc
            noteCode = str_to_imgCode_F4(currStr)
            if noteCode == currImg:
                solved = np.vstack([[list_img_idx[0], list_str_idx[0],0],LongestContinuousF4_recur(list_str[1:], list_img_str[1:],list_str_idx[1:], list_img_idx[1:], solutions)])
            elif (noteCode.startswith('n') and currImg.startswith('s')):
                solved = np.vstack([[list_img_idx[0], list_str_idx[0],0],LongestContinuousF4_recur(list_str, list_img_str[1:],list_str_idx, list_img_idx[1:], solutions)])
            else:
                notenum = int(noteCode[1:])
                opt3 = np.empty((0,3), dtype=int)
                if noteCode.startswith('n') and (notenum>12 or notenum<-2):
                    if (currImg == 'n'+str(notenum+1)) or (currImg == 'n'+str(notenum-1)):
                        opt3 = np.vstack([[list_img_idx[0], list_str_idx[0],1],LongestContinuousF4_recur(list_str[1:], list_img_str[1:],list_str_idx[1:], list_img_idx[1:], solutions)])
                assert len(list_img_idx) == len(list_img_str)
                opt1 = LongestContinuousF4_recur(list_str[1:], list_img_str,list_str_idx[1:], list_img_idx, solutions)
                opt2 = LongestContinuousF4_recur(list_str, list_img_str[1:],list_str_idx, list_img_idx[1:],solutions)
                lst = [len(opt1), len(opt2),len(opt3)]
                maxIdx = [i for i,v in enumerate(lst) if v==max(lst)]
                if len(maxIdx)==1:
                    chosen = maxIdx[0]
                else:
                    # TODO
                    logger.debug(f"there are {len(maxIdx)} options with same length")
                    lstsum = [np.sum(opt1.T[2]), np.sum(opt2.T[2]), np.sum(opt3.T[2])]
                    # filtered_lst is the number of "displacement sum" of those in maxIdx
                    filtered_lst = [lstsum[i] for i in maxIdx]
                    min_filt_idx = [i for i,v in enumerate(filtered_lst) if v==min(filtered_lst)]
                    chosen = maxIdx[min_filt_idx[0]]
                if chosen == 0:
                    solved = opt1
                elif chosen == 1:
                    solved = opt2
                else:
                    solved = opt3
    # key signature
    elif currStrTyp.startswith('k'):
        if currImg.startswith('c') or currImg.startswith('s'):
            # don't remove the str cause it might still be the next one's class
            solved = np.vstack([[list_img_idx[0], list_str_idx[0],0],LongestContinuousF4_recur(list_str, list_img_str[1:],list_str_idx, list_img_idx[1:], solutions)])
        else:
            # remove the str and keep the image?
            opt1 = LongestContinuousF4_recur(list_str[1:], list_img_str,list_str_idx[1:], list_img_idx, solutions)
            opt2 = LongestContinuousF4_recur(list_str, list_img_str[1:],list_str_idx, list_img_idx[1:], solutions)
            if len(opt1)>len(opt2):
                solved = opt1
            elif len(opt1)==len(opt2):
                if np.sum(opt1.T[2]) > np.sum(opt2.T[2]):
                    solved = opt2
                else:
                    solved = opt1
            else:
                solved = opt2
    # clef and barline
    elif currStrTyp.startswith('c') or currStrTyp.startswith('b'):
        if currImg.startswith(currStrTyp[0]):
            solved = np.vstack([[list_img_idx[0], list_str_idx[0],0],LongestContinuousF4_recur(list_str[1:], list_img_str[1:],list_str_idx[1:], list_img_idx[1:], solutions)])
        else:
            opt1 = LongestContinuousF4_recur(list_str[1:], list_img_str,list_str_idx[1:], list_img_idx, solutions)
            opt2 = LongestContinuousF4_recur(list_str, list_img_str[1:],list_str_idx, list_img_idx[1:], solutions)
            if len(opt1)>len(opt2):
                solved = opt1
            elif len(opt1)==len(opt2):
                if np.sum(opt1.T[2]) > np.sum(opt2.T[2]):
                    solved = opt2
                else:
                    solved = opt1
            else:
                solved = opt2
    else:
        logger.debug(currStrTyp)
        # only will have this if it's a time signature, move on for now (skip the time signature)
        # n for nonote
        assert (currStrTyp[0] == 't' or currStrTyp[0] == 'n')
        solved = LongestContinuousF4_recur(list_str[1:], list_img_str,list_str_idx[1:], list_img_idx, solutions)
    solutions[inputs] = solved
    return solved


# format element list to strings for processing
# notes: n12, n-1, ...etc, the position on staff
# rest: just r0
# clef: c1 for gclef(vln), c2 for fclef(cello)
# barline: just b
# sfn: just s
def reformat_element_list(elm_list:List[OmrStaff.StaffElements])->List[str]:
    formatted_str_list = []#[elm.typ[0]+str(elm.addval) for elm in elm_list] 
    for elm in elm_list:
        if elm.typ=='note' or elm.typ == 'clef':
            formatted_str_list.append(elm.typ[0]+str(elm.addval))
        elif elm.typ == 'rest':
            formatted_str_list.append(elm.typ[0]+'0')
        else:
            formatted_str_list.append(elm.typ[0])
    return formatted_str_list

def save_pred_img_G2(elements: List[OmrStaff.StaffElements], image, output_path = ''):
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
                elif 'thirty_second' in elm.newLabel:
                    color = (204,171,224)
                else:
                    logger.debug(elm.newLabel)
                    color = (0,150,0)
                img = cv2.rectangle(img, (elm.bbox[0], elm.bbox[1]), (elm.bbox[2],elm.bbox[3]), color, 2)
                img = cv2.putText(img, elm.newLabel[5:8], (elm.bbox[0], elm.bbox[1]-5) , cv2.FONT_HERSHEY_SIMPLEX,  0.7, color, 2, cv2.LINE_AA)
            else: 
                img = cv2.rectangle(img, (elm.bbox[0], elm.bbox[1]), (elm.bbox[2],elm.bbox[3]), (255,0,0), 2)
                img = cv2.putText(img, elm.newLabel[0:3], (elm.bbox[0], elm.bbox[1]-5) , cv2.FONT_HERSHEY_SIMPLEX,  0.7, (255,0,0), 2, cv2.LINE_AA)
        else:
            if elm.typ == 'note':
                color = (90,190,245)
                img = cv2.rectangle(img, (elm.bbox[0], elm.bbox[1]), (elm.bbox[2],elm.bbox[3]), color, 2)
                img = cv2.putText(img, imgCode_to_str_G2('n'+str(elm.addval)), (elm.bbox[0], elm.bbox[1]-5) , cv2.FONT_HERSHEY_SIMPLEX,  0.7, color, 2, cv2.LINE_AA)
    if output_path.endswith('.jpg'):
        cv2.imwrite(output_path, img)
    return img
def save_pred_img_F4(elements: List[OmrStaff.StaffElements], image, output_path = ''):
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
                elif 'thirty_second' in elm.newLabel:
                    color = (204,171,224)
                else:
                    logger.debug(elm.newLabel)
                    color = (0,150,0)
                img = cv2.rectangle(img, (elm.bbox[0], elm.bbox[1]), (elm.bbox[2],elm.bbox[3]), color, 2)
                img = cv2.putText(img, elm.newLabel[5:8], (elm.bbox[0], elm.bbox[1]-5) , cv2.FONT_HERSHEY_SIMPLEX,  0.7, color, 2, cv2.LINE_AA)
            else: 
                img = cv2.rectangle(img, (elm.bbox[0], elm.bbox[1]), (elm.bbox[2],elm.bbox[3]), (255,0,0), 2)
                img = cv2.putText(img, elm.newLabel[0:3], (elm.bbox[0], elm.bbox[1]-5) , cv2.FONT_HERSHEY_SIMPLEX,  0.7, (255,0,0), 2, cv2.LINE_AA)
        else:
            if elm.typ == 'note':
                color = (90,190,245)
                img = cv2.rectangle(img, (elm.bbox[0], elm.bbox[1]), (elm.bbox[2],elm.bbox[3]), color, 2)
                img = cv2.putText(img, imgCode_to_str_F4('n'+str(elm.addval)), (elm.bbox[0], elm.bbox[1]-5) , cv2.FONT_HERSHEY_SIMPLEX,  0.7, color, 2, cv2.LINE_AA)
    if output_path.endswith('.jpg'):
        cv2.imwrite(output_path, img)
    return img

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
                elif 'thirty_second' in elm.newLabel:
                    color = (204,171,224)
                else:
                    logger.debug(elm.newLabel)
                    color = (0,150,0)
                img = cv2.rectangle(img, (elm.bbox[0], elm.bbox[1]), (elm.bbox[2],elm.bbox[3]), color, 2)
                img = cv2.putText(img, elm.newLabel[5:8], (elm.bbox[0], elm.bbox[1]-5) , cv2.FONT_HERSHEY_SIMPLEX,  0.7, color, 2, cv2.LINE_AA)
            else: 
                img = cv2.rectangle(img, (elm.bbox[0], elm.bbox[1]), (elm.bbox[2],elm.bbox[3]), (255,0,0), 1)
                img = cv2.putText(img, elm.newLabel[0:3], (elm.bbox[0], elm.bbox[1]-5) , cv2.FONT_HERSHEY_SIMPLEX,  0.7, (255,0,0), 2, cv2.LINE_AA)
    if output_path.endswith('.jpg'):
        cv2.imwrite(output_path, img)
    return img

def add_label_to_element(elem_lst: List[OmrStaff.StaffElements], str_lst: List[str], img_str_map: np.array):
    for idxMap in img_str_map:
        elem_lst[idxMap[0]].addLabel(str_lst[idxMap[1]])
        elem_lst[idxMap[0]].setFlag(str_lst[idxMap[2]])

def runModel4(npy_path, pkl_path, img_path, output_dir):
    with open(pkl_path, 'rb') as file:
        staffList:List[OmrStaff] = pickle.load(file)
    pred_list = np.load(npy_path,allow_pickle=True).tolist()
    pred_list_split = [oneline.split('+') for oneline in pred_list]
    assert len(staffList) == len(pred_list_split)
    image = cv2.imread(img_path)
    img = image.copy()
    img_withadd = image.copy()
    for i in range(len(staffList)):
        curr_str_lst = pred_list_split[i]
        curr_img_lst = reformat_element_list(staffList[i].elements)
        if curr_str_lst[0] == 'clef-F4':
            img_str_map = LongestContinuousF4_recur(curr_str_lst, curr_img_lst,list(range(len(curr_str_lst))), list(range(len(curr_img_lst))),{})
            add_label_to_element(staffList[i].elements, curr_str_lst, img_str_map)
            img = save_pred(staffList[i].elements, img, '')
            img_withadd = save_pred_img_F4(staffList[i].elements, img_withadd, '')
        else:
            if not curr_str_lst[0] == 'clef-G2':
                logger.warn(f"curr_str_lst, i = {i} doen't start with any clef, default to G2(soprano)")
            img_str_map = LongestContinuousG2_recur(curr_str_lst, curr_img_lst,list(range(len(curr_str_lst))), list(range(len(curr_img_lst))),{})
            add_label_to_element(staffList[i].elements, curr_str_lst, img_str_map)
            img = save_pred(staffList[i].elements, img, '')
            img_withadd = save_pred_img_G2(staffList[i].elements, img_withadd, '')
            
    cv2.imwrite(output_dir+'mappingAllInOne.jpg', img)
    cv2.imwrite(output_dir+'mappingAllInOneWithAdd.jpg', img_withadd)
    logger.info(output_dir)
    return ''