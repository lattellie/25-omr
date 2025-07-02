import cv2
import glob 
import os
import numpy as np
from scipy import stats
from omr.part1 import runModel1
from omr.staffline_extraction import staff_extract_staffobj
from omr.part2 import runModel2
from omr import layers
from typing import List,Tuple, Union, Set
import statistics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
from get_prediction_singleStem import Single_Stem_Classifier
from omr.bbox import (
    BBox,
    merge_nearby_bbox,
)
import json
import math
from get_prediction import Sfn_Clef_classifier
from get_prediction_rest import Rest_Classifier
from abc import ABC, abstractmethod
from music21 import stream, note, metadata, chord, meter, clef, key
from fractions import Fraction
from dd_classes import RestNg
from tune_bar import tuneBar
from collections import Counter
from pdf2png import savePdf2Png
import pickle
import itertools

OUTPUT_BASE_FOLDER = 'string_dataset/output/beethoven'

NUM_TRACK = 4
TRACK_SHIFT = [0,0,0,0]
CLEF_OPTIONS = [[1],[1],[0,1],[-1,-2,1]]
RESIZE_RATIO_PACKED = 2
TIME_SIGNATURE = [9,8]
DEBUGIMG = None
class Staff:
    def __init__(self, left:int, right:int, ys:Tuple[int,int,int,int,int], minMaxDiff:int=0):
        self.left = left
        self.right = right
        self.ys = ys
        self.minDiff = minMaxDiff
    def get_yOne(self)->int:
        x = [self.ys[i+1] - self.ys[i] for i in range(len(self.ys) - 1)]
        return sum(x)//len(x)
    def get_yOne_float(self)->float:
        x = [self.ys[i+1] - self.ys[i] for i in range(len(self.ys) - 1)]
        return sum(x)/len(x)
    def IsStaffAligned(self, barheight:int)->bool:
        return self.minDiff<barheight/3
class Stem:
    def __init__(self, start:Tuple[int,int], isup:bool, notebbox:Tuple[int,int,int,int], smallBox:Tuple[int,int,int,int],hasStem:bool=True, alterbox: List[Tuple[int,int,int,int]]=[]):
        # start: x,y
        self.start:Tuple[int,int] = start
        self.hasStem:bool = hasStem
        self.isup:bool = isup
        # the x0,y0,x1,y1 of the alternative box (either go leftup or rightdown)
        self.alternativeBox: List[Tuple[int,int,int,int]] = alterbox
        self.end:Tuple[int,int] | None = None
        # the y0 to y1 of the left beam (y0 < y1)
        self.leftBeam: Tuple[int,int] = [start[1], start[1]]
        # the y0 to y1 of the right beam (y0 < y1)
        self.rightBeam: Tuple[int,int] = [start[1], start[1]]
        # -1: whole note, -2: 1/2, 0: 1/4, 1:1/8, 2:1/16 ... (the number of lines)
        self.rhythm: int = -1
        self.noteBox: Tuple[int,int,int,int] = notebbox
        self.isSingle: bool = False
        self.smallNoteBox: Tuple[int,int,int,int] = smallBox
        self.isOrnament: bool = False
        self.hasdot:bool = False
        self.hasLineMiddle:bool|None = None
        # 0 for the center of the staff (B in violin), 1 for C, -1 for A etc
        self.pitchFloat:float|None = None
        self.pitchWideFloat:float|None = None
        self.pitchSoprano: int|None = None
        self.accidentals: int|None = None
        self.accidentalBox: Tuple[int,int,int,int]|None = None
        # 0 for natural, 1 for sharp, -1 for flat
    def getBestPitchInt(self)->int:
        # if hasLineMiddle the number has to be even
        if self.hasLineMiddle is None:
            return round(self.pitchWideFloat)
        upperInt:int = math.ceil(self.pitchWideFloat)
        lowerInt:int = math.floor(self.pitchWideFloat)
        if upperInt == lowerInt:
            return upperInt
        if self.hasLineMiddle:
            return upperInt if upperInt%2==0 else lowerInt
        else:
            return upperInt if upperInt%2==1 else lowerInt

    def getX(self):
        return self.start[0]
    def setYlen(self, ylen:int):
        x = self.start[0]
        if self.isup:
            self.end = (x,self.start[1]-abs(ylen))
        else:
            self.end = (x,self.start[1]+abs(ylen))
    def getTopCoord(self):
        if self.isup:
            return self.end
        else:
            return self.start
    def getBottomCoord(self):
        if self.isup:
            return self.start
        else:
            return self.end
    def getY0Y1(self):
        return (min(self.start[1],self.end[1]), max(self.start[1],self.end[1]))
    def getLineX0Y0X1Y1(self, width:int=3):
        return (self.start[0]-width//2, self.getTopCoord()[1], self.start[0]+(width-width//2), self.getBottomCoord()[1])
    def getYLen(self)->int:
        return int(abs(self.start[1]-self.end[1]))
    def setBeam(self, typ:str,top:int|None = None, bottom:int|None = None):
        if typ.lower().startswith('l'):
            self.setLeftBeam(top=top, bottom=bottom)
        else:
            self.setRightBeam(top=top, bottom=bottom)
    def setLeftBeam(self, top:int|None = None, bottom:int|None = None):
        if top is not None:
            self.leftBeam[0] = top
        if bottom is not None:
            self.leftBeam[1] = bottom
    def setRightBeam(self, top:int|None = None, bottom:int|None = None):
        if top is not None:
            self.rightBeam[0] = top
        if bottom is not None:
            self.rightBeam[1] = bottom 
    def getBeamHeights(self):
        return (self.leftBeam[1]-self.leftBeam[0], self.rightBeam[1]-self.rightBeam[0])
    def getBoundingBox(self):
        x0,y0,x1,y1 = self.noteBox
        yy0 = self.start[1]
        yy2 = self.end[1]
        return (x0,min(y0,yy0,yy2), x1, max(y0,yy0,yy2))
    def getString(self):
        keyName = ['C','D','E','F','G','A','B']
        currKeyName = keyName[(self.pitchSoprano-1)%7]
        currKey = currKeyName+str((self.pitchSoprano-1)//7+5)
        if self.accidentals is not None: # it has keySignature
            if self.accidentals == -1:
                currKey = currKey[0]+'-'+currKey[1]
            elif self.accidentals == 1:
                currKey = currKey[0]+'#'+currKey[1]
        return currKey

class Rest:
    def __init__(self, bbox:Tuple[int,int,int,int], rhythm: int):
        self.boundingBox:Tuple[int,int,int,int] = bbox
        # rest -1: whole rest or 1/2 depending, 0: 1/4, 1:1/8, 2:1/16 ... (the number of lines)
        self.rhythm: int = rhythm
        self.hasdot: bool = False
        self.noteGroupId: int|None = None
        self.tunedLength: Fraction|None = None
    def setNgId(self, ngId:int):
        self.noteGroupId = ngId
    def getLengthFraction(self):
        restClassNames = ['1/4','1/8','1/16','1/32','1/2'] # only for >=0
        ratio = Fraction(3,2) if self.hasdot else 1
        return Fraction(restClassNames[self.rhythm])*ratio
    def getString(self):
        return f"Rest_{self.tunedLength}"

class NoteGroup:
    def __init__(self, noteStemObj:Stem):
        # the width of the stem box we want
        self.stemWidth:int = 3
        # x0,y0,x1,y1
        self.noteBoxes:List[Tuple[int,int,int,int]] = [noteStemObj.noteBox]
        # x0, y0, x1, y1
        self.stemLineBoxes:List[Tuple[int,int,int,int]] = [noteStemObj.getLineX0Y0X1Y1(width=self.stemWidth)]
        # x0,y0,x1,y1
        self.boundingBox: Tuple[int,int,int,int]|None = None
        self.noteStemList: List[Stem] = [noteStemObj]
        self.updateBoundingBox()
        self.noteChunkId: int|None = None
        # self.pitchLists: List[float] = []
        # for instance C has line, D doesn't
        # self.noteLineMiddle: List[bool] = []
        self.restList:List[Rest] = []
        self.tunedLength: Fraction|None = None
    def addRest(self, rest:Rest):
        self.restList.append(rest)
        self.updateBoxRest(rest.boundingBox)
    def addNoteStem(self, noteStemObj: Stem):
        self.noteStemList.append(noteStemObj)
        self.noteBoxes.append(noteStemObj.noteBox)
        self.stemLineBoxes.append(noteStemObj.getLineX0Y0X1Y1(width=self.stemWidth))
        self.updateBoundingBox()
    def updateBoundingBox(self):
        noteMinX = min([n[0] for n in self.noteBoxes])
        noteMaxX = max([n[2] for n in self.noteBoxes])
        noteMinY = min([n[1] for n in self.noteBoxes])
        noteMaxY = max([n[3] for n in self.noteBoxes])
        stemMinX = min([n[0] for n in self.stemLineBoxes])
        stemMaxX = max([n[2] for n in self.stemLineBoxes])
        stemMinY = min([n[1] for n in self.stemLineBoxes])
        stemMaxY = max([n[3] for n in self.stemLineBoxes])
        self.boundingBox = (min(noteMinX, stemMinX), min(noteMinY, stemMinY), max(noteMaxX, stemMaxX), max(noteMaxY, stemMaxY))
    def updateBoxRest(self, restBox:Tuple[int,int,int,int]):
        minx,miny,maxx,maxy = self.boundingBox
        x0,y0,x1,y1 = restBox
        self.boundingBox = (min(minx, x0), min(miny, y0), max(maxx, x1), max(maxy, y1))
    def getMinLength(self):
        class_names = ['1/4','1/8','1/16','1/32', '1/64', '1/128', '1/2', '1/1']
        minLength = 1
        for ns in self.noteStemList:
            if Fraction(class_names[ns.rhythm])*(1.5 if ns.hasdot else 1)<minLength:
                minLength = Fraction(class_names[ns.rhythm])*(Fraction(3,2) if ns.hasdot else 1)
        return minLength
    def getString(self):
        stemStrings = ""
        for ns in self.noteStemList:
            stemStrings+=ns.getString()
            stemStrings+="|"
        return f"Note_{stemStrings}{self.tunedLength}"

def mergeNoteGroup(ng1:NoteGroup, ng2: NoteGroup, beamMapImg: np.ndarray):
    thres = 0.3
    nsl1:List[Stem] = ng1.noteStemList
    nslSum1 = [(np.sum(beamMapImg[sm.noteBox[1]:sm.noteBox[3],sm.noteBox[0]:sm.noteBox[2],0]==255))/((sm.noteBox[2]-sm.noteBox[0])*(sm.noteBox[3]-sm.noteBox[1]))for sm in nsl1]    
    nsl2:List[Stem] = ng2.noteStemList
    nslSum2 = [(np.sum(beamMapImg[sm.noteBox[1]:sm.noteBox[3],sm.noteBox[0]:sm.noteBox[2],0]==255))/((sm.noteBox[2]-sm.noteBox[0])*(sm.noteBox[3]-sm.noteBox[1]))for sm in nsl2]
    # case one: both are single notes -> filter out the one with greater beams value and >0.5
    if len(nsl1) == 1 and len(nsl2) ==1:
        if nslSum1[0] > nslSum2[0]*2 and nslSum1[0]>thres:
            return ng2
        elif nslSum2[0] > nslSum1[0]*2 and nslSum2[0]>thres:
            return ng1
        else:
            ng1.addNoteStem(ng2.noteStemList[0])
            return ng1
    # if one has more than one notes -> check the other one
    else:
        if len(nsl1)>len(nsl2):
            if nslSum2[0]>thres:
                return ng1
            elif nslSum1[0]>nslSum2[0]*2 and nslSum1[0]>thres:
                return ng2
        else:
            if nslSum1[0]>thres:
                return ng2
            elif nslSum2[0]>nslSum1[0]*2 and nslSum2[0]>thres:
                return ng1
    for ns in nsl2:
        ng1.addNoteStem(ns)
    return ng1

class NoteChunk:
    def __init__(self, noteGroups:Set[int]):
        self.noteGroupIdxs:Set[int] = noteGroups
    def mergeNoteChunk(self, noteGroup2: Set[int]):
        self.noteGroupIdxs:Set[int] = self.noteGroupIdxs.union(noteGroup2)
    def getBoundingBox(self,noteGroupList:List[NoteGroup]):
        x0s = min([noteGroupList[i].boundingBox[0] for i in self.noteGroupIdxs])
        x1s = max([noteGroupList[i].boundingBox[2] for i in self.noteGroupIdxs])
        y0s = min([noteGroupList[i].boundingBox[1] for i in self.noteGroupIdxs])
        y1s = max([noteGroupList[i].boundingBox[3] for i in self.noteGroupIdxs])
        return [x0s, y0s, x1s, y1s]
class sfnClefInterface(ABC):
    @abstractmethod
    def getString(self):
        pass
    @abstractmethod
    def getBbox(self) -> Tuple[int,int,int,int]:
        pass
    @abstractmethod
    def getValue(self) -> int:
        pass
    @abstractmethod
    def getType(self) -> int:
        # 0 is accidental, 1 is clef
        pass
class Accidentals(sfnClefInterface):
    def __init__(self, bbox:Tuple[int,int,int,int], shift: int):
        self.boundingBox:Tuple[int,int,int,int] = bbox
        # shift is 1 for sharp, 0 for natural, -1 for flat
        self.shift: int = shift
        self.isKeySignature:bool =  False
        self.endKeySignature:bool = False
        self.ksKeySop: float|None = None
        self.shrinkYs: Tuple[int,int]|None = None
        self.ngIndex: int|None = None
    def getString(self):
        if self.shift == 0:
            return 'natural'
        elif self.shift == 1:
            return 'sharp'
        elif self.shift == -1:
            return 'flat'
        else:
            return 'invalid accidentals'
    def getColor(self):
        clrs =  [(255,255,0),(255,0,125),(255,0,255)]
        return clrs[self.shift]
    def getBbox(self) -> Tuple[int]:
        return self.boundingBox
    def getValue(self) -> int:
        return self.shift
    # 0 if is accidentals
    def getType(self)->int:
        return 0
    def getSign(self)->int:
        if self.shift == 0:
            return '%'
        elif self.shift == 1:
            return '#'
        elif self.shift == -1:
            return 'b'
        else:
            return 'invalid accidentals'
class Clef(sfnClefInterface):
    def __init__(self, bbox:Tuple[int,int,int,int], type: int):
        self.boundingBox:Tuple[int,int,int,int] = bbox
        self.type:int = type
        # how far it should shift if the staff position is 0 in the sidebar
    def getString(self):
        if self.type == 1:
            return 'violin'
        elif self.type == 0:
            return 'viola'
        elif self.type == -1:
            return 'bass'
        elif self.type == -2:
            return 'violaCello'
        else:
            return 'invalid clef'
    def getBbox(self) -> Tuple[int]:
        return self.boundingBox
    def getValue(self) -> int:
        return self.type
    # 1 if it's clef
    def getType(self)->int:
        return 1
    

class Bar:
    def __init__(self, TS:Tuple[int,int]=TIME_SIGNATURE):
        self.ts = TS
        self.elementList:List[Accidentals|Clef|Rest|NoteGroup] = []
        self.restNgList:List[RestNg] = []
    def getString(self):
        retStr = []
        for elem in self.elementList:
            retStr.append(elem.getString())   
        return retStr  
    def addElement(self, elem:Union[Accidentals, Clef, Rest, NoteGroup]):
        if type(elem) == NoteGroup:
            if elem.noteChunkId is not None and not False in [j.rhythm<=0 for j in elem.noteStemList]:
                print(f'removing element in noteChunk {elem.noteChunkId} > quarter')
                return
        self.elementList.append(elem)
    def getTunedTotalBeat(self) -> Fraction:
        bt = 0
        for elm in self.elementList:
            if type(elm) == NoteGroup:
                bt += elm.tunedLength
            elif type(elm) == Rest:
                bt += elm.tunedLength
        return bt

    def getTotalBeat(self)->Fraction:
        bt = 0
        for elm in self.elementList:
            if type(elm) == NoteGroup:
                bt += elm.getMinLength()
            elif type(elm) == Rest:
                bt += elm.getLengthFraction()
        return bt
    def getRhythmList(self) -> List[Fraction]:
        retLst = []
        for elm in self.elementList:
            if type(elm) == NoteGroup:
                retLst.append(elm.getMinLength())
            elif type(elm) == Rest:
                retLst.append(elm.getLengthFraction())
        return retLst
    def getRestNg(self) -> List[RestNg]:
        if len(self.restNgList)!= 0:
            print("returning restNg list from last time")
            return self.restNgList
        retLst = []
        for elm in self.elementList:
            if type(elm) == NoteGroup:
                groupId = 0
                if elm.noteChunkId is not None:
                    groupId = elm.noteChunkId
                beamLength = (-1,-1)
                beamEnd = [-1,-1] # for grouped notes
                length = elm.getMinLength()
                if len(elm.noteStemList) == 1:
                    currStm = elm.noteStemList[0]
                    beamLength = [int(a) for a in currStm.getBeamHeights()]
                    if currStm.isup:
                        beamEnd = [currStm.getX(), int(min(currStm.getY0Y1()))]
                    else:
                        beamEnd = [currStm.getX(), int(max(currStm.getY0Y1()))]
                newElm = RestNg(length=Fraction(length),
                                groupId = groupId,
                                isRest = False,
                                numNotes = len(elm.noteStemList),
                                beamLength = beamLength,
                                beamEnd=beamEnd)
                retLst.append(newElm)
            elif type(elm) == Rest:
                newElm = RestNg(length=elm.getLengthFraction()*Fraction(3,2) if elm.hasdot else elm.getLengthFraction(),
                                groupId = -1,
                                isRest = True,
                                numNotes = 0,
                                beamLength = (-1,-1),
                                beamEnd=[-1,-1])
                retLst.append(newElm)
        self.restNgList = retLst
        return retLst
    def reassignLength(self, origLens: List[Fraction], newLens: List[Fraction]):
        currNewLenIdx = 0
        for elm in self.elementList:
            if type(elm)!=NoteGroup and type(elm)!=Rest:
                continue
            if newLens[currNewLenIdx] is not None:
                elm.tunedLength = newLens[currNewLenIdx]
            else:
                elm.tunedLength = origLens[currNewLenIdx]
            currNewLenIdx+=1
print_str = True
def printt(str):
    if print_str:
        print(str)

debugImg = False
# def imwrite(str,img,strictlyYes=False, startingFolder = 'test_images/'):
def imwrite(str,img,strictlyYes=False, startingFolder = f'{OUTPUT_BASE_FOLDER}/'):
# def imwrite(str,img,strictlyYes=False, startingFolder = ''):
    if debugImg or strictlyYes:
        print(f'saving {str}')
        cv2.imwrite(f'{startingFolder}debug/{str}',img)
# def outputImWrite(str,img,rootFolder='test_images/output'):
# def outputImWrite(str,img,rootFolder='output_images'):
def outputImWrite(str,img):
    cv2.imwrite(f'{OUTPUT_BASE_FOLDER}/output/{str}',img)

# barline_height = 12

def getStemBox(dataDict, missing_height = 33):
    stems_rests = dataDict['stems_rests']
    clefs_keys = dataDict['clefs_keys']
    notehead = dataDict['notehead']
    staff = dataDict['staff']
    symbols = dataDict['symbols']
    image = dataDict['image']

    _, img128 = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5),dtype=np.uint8)
    beam = cv2.erode(img128, kernel, iterations = 1)
    imwrite('beam.jpg',beam)

    combine = beam.copy()
    notehead_clefkey = notehead+clefs_keys
    xs, ys = np.where(notehead_clefkey>0)
    combine[xs,ys] = 0
    xs, ys = np.where(symbols>0)
    combine[xs,ys] = 255
    combine = cv2.bitwise_not(combine)
    imwrite('combine.jpg',combine)
    
    stem = np.zeros_like(img128)
    xs, ys = np.where(stems_rests>0)
    stem[xs,ys] = 255
    stem = cv2.erode(stem, np.ones((3,1),dtype=np.uint8),iterations = 1)
    stem = cv2.bitwise_not(stem)
    imwrite('stem.jpg',stem)

    note_stem = notehead+stems_rests
    notestem = np.zeros_like(img128)
    xs,ys = np.where(note_stem>0)
    notestem[xs,ys] = 255
    notestem = cv2.erode(notestem, np.ones((3,1),dtype=np.uint8),iterations = 1)
    notestem = cv2.bitwise_not(notestem)
    imwrite('notestem.jpg',notestem)

    ret,mod_stem = cv2.threshold(cv2.cvtColor(stem, cv2.COLOR_BGR2GRAY), 128,255,cv2.THRESH_BINARY_INV)
    # now 255 is where the stems etc are

    imwrite('mod_stem.jpg',mod_stem)
    # remove circles (the rests)
    circle_size = 6
    remove = cv2.erode(mod_stem, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(circle_size,circle_size)), iterations=1)
    remove = cv2.dilate(remove, cv2.getStructuringElement(cv2.MORPH_RECT,(circle_size*4,circle_size*6)), iterations=1)

    # remove long beams
    remove1 = cv2.dilate(mod_stem, np.ones((1,3),dtype=np.uint8),iterations = 1)
    remove1 = cv2.erode(remove1, np.ones((1,7),dtype=np.uint8), iterations=1)
    remove1 = cv2.dilate(remove1,np.ones((3,9),dtype=np.uint8), iterations=1)

    mod_stem[remove>0]=0
    mod_stem[remove1>0] = 0
    imwrite('test_remove1.jpg',remove1)
    imwrite('test_remove.jpg',remove)
    imwrite('test_mod.jpg',mod_stem)

    
    # remove noise
    single_kernel = np.array([
        [0,255,0],
        [0,255,0],
    ], dtype=np.uint8)
    mod = cv2.erode(mod_stem, single_kernel, iterations = 1)
    mod = cv2.dilate(mod, single_kernel, iterations = 1)
    imwrite('test_mod2.jpg',mod)
    
    currLen = missing_height
    while currLen>=3:
        kernel = np.zeros((currLen,1),dtype=np.uint8)
        kernel[0]=1
        kernel[currLen-1]=1
        add = cv2.dilate(mod, kernel, iterations=1)
        mod = mod+add
        mod[mod>255] = 255
        currLen = currLen//2+1

    imwrite('test.jpg',cv2.bitwise_not(mod))
    xs,ys = np.where(mod>0)
    imgrgb_ori = combine
    imgrgb = imgrgb_ori.copy()
    assert imgrgb.shape[0] == stem.shape[0]
    assert imgrgb.shape[1] == stem.shape[1]
    imgrgb[xs,ys] = (0,0,255)
    imwrite('add_imgrgb.jpg',imgrgb)
    imwrite('stemLocation_mod.jpg',mod)
    return image, imgrgb_ori, mod

def get_bbox(data: np.ndarray) -> List[Tuple[int,int,int,int]]:
    contours, _ = cv2.findContours(data.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box = (x, y, x+w, y+h)
        bboxes.append(box)
    return bboxes

def filter_small_bbox_area(data: np.ndarray, noteimg:np.ndarray, xmin, ymin, areamin):
    filtered_Box = []
    for box in data:
        x0,y0,x1,y1 = box
        w = x1-x0
        h = y1-y0
        if w>=xmin and h>=ymin and w*h>areamin:
            filtered_Box.append(box)
        else:
            noteimg[y0:y1,x0:x1] = 0
    return filtered_Box,noteimg

def filter_small_bbox(data: np.ndarray, xmin, ymin):
    filtered_Box = []
    for box in data:
        x0,y0,x1,y1 = box
        w = x1-x0
        h = y1-y0
        if w>=xmin and h>=ymin:
            filtered_Box.append(box)
    return filtered_Box

def staffs_to_omrStaffList(staffs):
    left = min([sf.x_left for sf in staffs[0,:]])
    right = max([sf.x_right for sf in staffs[-1,:]])
    omrstaff_list: List[Staff] = []
    staffRange = []
    mindiffs = []
    maxdiffs = []
    for i in range(staffs.shape[1]):
        yuppers = [sf.y_upper for sf in staffs[:,i]]
        ybottoms = [sf.y_lower for sf in staffs[:,i]]
        top = statistics.median(yuppers)
        bottom = statistics.median(ybottoms)
        unit_size = statistics.median([sf.unit_size for sf in staffs[:,i]])
        minMaxDiff = min(max(yuppers)-min(yuppers), max(ybottoms)-min(ybottoms))
        maxMaxDiff = max(max(yuppers)-min(yuppers), max(ybottoms)-min(ybottoms))
        ys = [int(top), int(top+unit_size), int((top+bottom)/2),int(bottom-unit_size),int(bottom)]
        sf = Staff(int(left), int(right), ys, minMaxDiff)
        mindiffs.append(minMaxDiff)
        maxdiffs.append(maxMaxDiff)
        omrstaff_list.append(sf)
        staffRange.append(range(int(top),int(bottom)))
    # print(f'MinDiff: {["{:.2f}".format(num) for num in mindiffs]}, average = {np.mean(mindiffs)}')
    # print(f'MaxDiff: {["{:.2f}".format(num) for num in maxdiffs]}, average = {np.mean(maxdiffs)}')
    # print('')
    return omrstaff_list, staffRange

def init_bar_height(dataDict, min_barheight = 8):
    staff = dataDict['staff']
    staffs, zones = staff_extract_staffobj(staff, min_barheight)
    staffList, staffRange = staffs_to_omrStaffList(staffs)
    barheight = [sf.get_yOne_float() for sf in staffList]
    avg_barheight = int(sum(barheight)/len(barheight))
    return avg_barheight, staffList

def split_connected_notes(bboxList, bar_height,image):
    minh = int(bar_height*0.9)
    minw = int(bar_height)
    retlst = []
    for bbox in bboxList:
        x0,y0,x1,y1 = bbox
        wrat = (x1-x0)/minw
        hrat = (y1-y0)/minh
        wnum = int(wrat)
        hnum = int(hrat)
        note_w = int(minw*1.3)
        note_h = int(minh*1.3)
        if wnum*hnum <= 1:
            retlst.append(bbox)
        elif wnum>=2: 
            if y1-y0<=bar_height or wnum>=3:
                xwid = (x1-x0)//wnum
                for i in range(wnum):
                    retlst.append((x0+i*xwid,y0,x0+(i+1)*xwid,y1))
            # it's a scatter of three vertically stacked notes with two adjacent notes
            elif hnum>2.5 and hnum>wnum:
                # first get the top and bottom ones
                # if the top part has more black on the left (x0)
                option2s = []
                if np.sum(image[y0:y0+note_h,x0:x0+note_w])>np.sum(image[y0:y0+note_h,x1-note_w:x1]):
                    retlst.append((x0,y0,x0+note_w, y0+note_h))
                    # the other side go down once
                    option2s.append((x1-note_w,y0+note_h//2,x1, y0+int(note_h*1.5)))
                else:
                    retlst.append((x1-note_w,y0,x1, y0+note_h))
                    option2s.append((x0,y0+note_h//2,x0+note_w, y0+int(note_h*1.5)))
                # if the bottom part has more black on the left (x0)
                if np.sum(image[y1-note_h:y1,x0:x0+note_w])>np.sum(image[y1-note_h:y1,x1-note_w:x1]):
                    retlst.append((x0,y1-note_h,x0+note_w, y1))
                    option2s.append((x1-note_w,y1-int(note_h*1.5),x1, y1-note_h//2))
                else:
                    retlst.append((x1-note_w,y1-note_h,x1, y1))
                    option2s.append((x0,y1-int(note_h*1.5),x0+note_w, y1-note_h//2))
                # deal with the middle one    
                if np.sum(image[option2s[0][1]:option2s[0][3], option2s[0][0]:option2s[0][2]]) > np.sum(image[option2s[1][1]:option2s[1][3], option2s[1][0]:option2s[1][2]]):
                    retlst.append(option2s[0])
                else:
                    retlst.append(option2s[1])
            else:
                # if the left part has more black on top(y0)
                if np.sum(image[y0:y0+note_h,x0:x0+note_w])>np.sum(image[y1-note_h:y1,x0:x0+note_w]):
                    retlst.append((x0,y0,x0+note_w, y0+note_h))
                else:
                    retlst.append((x0,y1-note_h,x0+note_w, y1))
                # if the right part has more black on top(y1)
                if np.sum(image[y0:y0+note_h,x1-note_w:x1])>np.sum(image[y1-note_h:y1,x1-note_w:x1]):
                    retlst.append((x1-note_w,y0,x1, y0+note_h))
                else:
                    retlst.append((x1-note_w,y1-note_h,x1, y1))

        elif hnum>=2:
            ywid = (y1-y0)//hnum
            for i in range(hnum):
                retlst.append((x0,y0+i*ywid,x1,y0+(i+1)*ywid))
    return retlst
    


def assignBeamLengthBeamImg(stem_list:List[Stem], image:np.ndarray, beam:np.ndarray, barheight:int):
    x_diff = barheight//2
    beam = cv2.cvtColor(beam, cv2.COLOR_BGR2GRAY)
    # if it's completely white in the white_buffer_max area then stop
    white_buffer_max = barheight
    beam_start_buffer_max = barheight//3
    for stem in stem_list:
        xCenter,yCenter = stem.start
        xLeft = xCenter-x_diff
        xRight = min(image.shape[1]-1,xCenter+x_diff)
        yMax = stem.getYLen()
        if stem.getYLen() == 0:
            continue
        if stem.isup:
            for xtype in ['left', 'right']:
                if xtype=='left':
                    currX = xLeft
                else:
                    currX = xRight
                currY = stem.getTopCoord()[1]
                # if the starting point of beam is not white, meaning that it might be a symbol, staff line etc
                # keep going down until it's at the right place
                if sum(np.max(beam[currY:currY+beam_start_buffer_max, xCenter-barheight//2:xCenter+barheight//2],1)) < 255*beam_start_buffer_max:
                    while(sum(np.max(beam[currY:currY+beam_start_buffer_max, xCenter-barheight//2:xCenter+barheight//2],1)) < 255*beam_start_buffer_max and currY<stem.getBottomCoord()[1]-bar_height//2):
                        currY = currY+1
                stem.end = (stem.end[0], currY)

                if beam[currY, currX] == 0:
                    while(beam[currY, currX] == 0 and currY>stem.getBottomCoord()[1]+bar_height//2):
                        currY = currY+1
                else:
                    while beam[currY-1, currX] == 255:
                        currY = currY-1
                # now the topStartY is the "real start"
                stem.setBeam(typ=xtype, top=currY)
                # go down until it's wholely black
                while sum(beam[currY:currY+white_buffer_max, currX])>255:
                    currY+=1
                stem.setBeam(typ=xtype, bottom=currY)
        else:
            for xtype in ['left', 'right']:
                if xtype=='left':
                    currX = xLeft
                else:
                    currX = xRight
                currY = stem.getBottomCoord()[1]

                # if the starting point of beam is not white, meaning that it might be a symbol, staff line etc
                # keep going up until it's at the right place
                if sum(np.max(beam[currY-beam_start_buffer_max:currY, xCenter-barheight//2:xCenter+barheight//2],1)) < 255*beam_start_buffer_max:
                    while(sum(np.max(beam[currY-beam_start_buffer_max:currY, xCenter-barheight//2:xCenter+barheight//2],1)) < 255*beam_start_buffer_max and currY>stem.getTopCoord()[1]+bar_height//2):
                        currY = currY-1
                stem.end = (stem.end[0], currY)

                # if it's black (need to go up)
                if beam[currY, currX] == 0:
                    while(beam[currY, currX] == 0 and currY<stem.getTopCoord()[1]-bar_height//2):
                        currY = currY-1
                else:
                    while beam[currY+1, currX] == 255:
                        currY = currY+1
                # now the currY is the "real start"
                stem.setBeam(typ=xtype, bottom=currY)
                # go up until it's wholely black
                while sum(beam[currY-white_buffer_max:currY, currX])>255:
                    currY-=1
                stem.setBeam(typ=xtype, top=currY)
    imgBgr = image.copy()
    beam_heights = []
    for stem in stem_list:
        beam_heights.append(stem.leftBeam[1]-stem.leftBeam[0])
        beam_heights.append(stem.rightBeam[1]-stem.rightBeam[0])
        imgBgr = cv2.rectangle(imgBgr, (stem.start[0]-x_diff, stem.leftBeam[0]), (stem.start[0]-x_diff, stem.leftBeam[1]),(0,0,255),3,cv2.LINE_AA)
        imgBgr = cv2.rectangle(imgBgr, (stem.start[0]+x_diff, stem.rightBeam[0]), (stem.start[0]+x_diff, stem.rightBeam[1]),(255,255,0),3, cv2.LINE_AA)
    imwrite('leftrightRectBeam.jpg', imgBgr)
    return imgBgr, stem_list, beam_heights

def save_beam_hist(beam_heights: List[int], img_name:str, barheight):
    plt.hist([beam_height/bar_height for beam_height in beam_heights], max(beam_heights))
    plt.xlim(0,3.5)
    plt.savefig(f'hist_{img_name}.jpg', format='jpg')        
    plt.close()
    count, _ = np.histogram(beam_heights, max(beam_heights))
    whole_thres = [1, int(barheight*0.6), int(barheight*1.5), int(barheight*2.7), int(barheight*3.2)]
    # expand the list to be able to access list[init_thres[-1]]
    init_thres = whole_thres[:sum([0 if thres>len(count) else 1 for thres in whole_thres])+1]
    expanded_count = list(count) + [0]*(init_thres[-1]+1-len(count))
    while sum(expanded_count[init_thres[-1]:])<max(len(beam_heights)//100, 5):
        init_thres.pop(-1)
    init_thres.append(len(expanded_count))
    centers = [0]*len(init_thres)
    for i in range(1,len(init_thres)):
        centers[i] = np.argmax(expanded_count[init_thres[i-1]:init_thres[i]])+init_thres[i-1]

def get_beam_height(beam_heights:List[int], barheight):
    whole_thres = [1, int(barheight*0.6), int(barheight*1.8), int(barheight*2.7), int(barheight*3.2)]
    if max(beam_heights) == 0:
        return beam_heights
    count, _ = np.histogram(beam_heights, max(beam_heights))
    # expand the list to be able to access list[init_thres[-1]]
    init_thres = whole_thres[:sum([0 if thres>len(count) else 1 for thres in whole_thres])+1]
    expanded_count = list(count) + [0]*(init_thres[-1]+1-len(count))
    while len(init_thres)>0 and sum(expanded_count[init_thres[-1]:])<max(len(beam_heights)//100,5):
        tooBig = init_thres.pop(-1)
        beam_heights = [tooBig if bh>tooBig else bh for bh in beam_heights]
        expanded_count[tooBig:] = [0]*len(expanded_count[tooBig:])
    init_thres.append(len(expanded_count))
    centers = [0]*len(init_thres)
    for i in range(1,len(init_thres)):
        centers[i] = np.argmax(expanded_count[init_thres[i-1]:init_thres[i]])+init_thres[i-1]
    kmeans = KMeans(n_clusters = len(centers), init = np.array(centers, dtype=np.uint8).reshape(-1, 1), n_init = 1)
    kmeans.fit(np.array(beam_heights,dtype=np.uint8).reshape(-1,1))
    return kmeans.labels_.tolist()

keyCharMap = {'z':'n2',
              'x': 'n4',
              'c': 'n8',
              'v': 'n16',
              's':'noClass',
              'b': 'n1'}
def init_label_folder(rootFolder = 'training'):
    for saveFolderName in ['StemUp', 'StemDown']:
        for saveFolderNameAdd in ['img200', 'beam']:
            folderPath = os.path.join(rootFolder,f'{saveFolderName}_{saveFolderNameAdd}')
            if not os.path.exists(folderPath):
                os.mkdir(folderPath)
            for one_dir in list(keyCharMap.values()):
                if not os.path.exists(os.path.join(folderPath,one_dir)):
                    os.mkdir(os.path.join(folderPath,one_dir))

def labelStemData(stem: Stem, img200bwInv: np.ndarray, beamImgBwInv: np.ndarray, idx:int=0, rootFolder = 'training') -> bool:
    sy0, sy1 = stem.getY0Y1()
    nx0, ny0, nx1, ny1 = stem.noteBox
    if stem.isup:
        subFolder='StemUp'
        x0 = nx0
        x1 = min(nx1+(nx1-nx0), img200bwInv.shape[1]-1)
        y0 = sy0
        y1 = ny1
    else:
        subFolder = 'StemDown'
        x0 = max(nx0-(nx1-nx0)//2,0)
        x1 = nx1
        y0 = ny0
        y1 = sy1
    img200Crop = img200bwInv[y0:y1, x0:x1]
    beamCrop = beamImgBwInv[y0:y1, x0:x1]
    assert beamCrop.shape[0] == beamCrop.shape[0]
    cv2.imshow(f'{idx}', np.hstack((img200Crop, beamCrop)))
    k=cv2.waitKey()
    cv2.destroyAllWindows()
    if chr(k) == 'a':
        print(f"stopping at img idx: {idx}")
        return False
    else:
        filname = f'{img_name}_{idx}.jpg'
        classFolder = 'x'
        key_chr = chr(k)
        try:
            classFolder = keyCharMap[key_chr]
        except:
            print(f'invalid keychr: {key_chr}')
        cv2.imwrite(f'{rootFolder}/{subFolder}_img200/{classFolder}/{filname}', img200Crop)
        cv2.imwrite(f'{rootFolder}/{subFolder}_beam/{classFolder}/{filname}', beamCrop)
        return True
def getSingleStemPred(img, model:Single_Stem_Classifier):
    # ['n2', 'n4', 'n816'] (0,1,2)if it's flat then it should be n1 3
    if img.shape[1]/img.shape[0]>1.5:
        return 3
    else:
        return model.predict(img)
def knnRhythmAndDraw(stem_list_assigned:List[Stem], beam_heights, image, bar_height, beamMapImg, stemUpClassifier, stemDownClassifier,img_name):
    # save_beam_hist(beam_heights, img_name, bar_height)
    beam_type = get_beam_height(beam_heights, bar_height)
    class_colors = [(255,0,0),(0,0,255),(255,255,0),(0,120,255),(40,255,40), (245, 220, 255),(230,130,175),(165,170, 70)]
    class_names = ['1/4','1/8','1/16','1/32', '1/64', '1/128', '1/2', '1/1']
    imgrgb = image.copy()
    x_diff = bar_height//2
    mapBeamVal, middle, mapStaffNum = cv2.split(beamMapImg)
    # the first one is a empty one just for easier indexing
    noteGroupStemMap = np.zeros((beamMapImg.shape[0], beamMapImg.shape[1]), dtype=np.uint16)
    noteGroupList:List[NoteGroup|None] = [None]
    img200 =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,img200bw = cv2.threshold(img200, 200, 255, cv2.THRESH_BINARY_INV)
    for j in range(0,len(class_names)):
        imgrgb = cv2.putText(imgrgb,class_names[j], (30, 30+j*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_colors[j], 2, cv2.LINE_AA)
    # stillLabel = True
    # init_label_folder()
    predLabelConvert = [-2,0,1,-1]
    for idx,stem in enumerate(stem_list_assigned):
        stem.rhythm = max(beam_type[idx*2: idx*2+2])
        leftHeight, rightHeight = stem.getBeamHeights()
        # solve the issue of misclassify something to no stem when there's one (since knn also train on 0s)
        if stem.rhythm==0 and max(leftHeight, rightHeight)>0:
            stem.rhythm = 1
        elif stem.rhythm == 0:
            sy0, sy1 = stem.getY0Y1()
            nx0, ny0, nx1, ny1 = stem.noteBox
            if stem.isup:
                x0 = nx0
                x1 = min(nx1+(nx1-nx0), img200bw.shape[1]-1)
                y0 = sy0
                y1 = ny1
            else:
                x0 = max(nx0-(nx1-nx0)//2,0)
                x1 = nx1
                y0 = ny0
                y1 = sy1
            img200Crop = img200bw[y0:y1, x0:x1]
            # ['n2', 'n4', 'n816'] (0,1,2), 3 if the box doesn't have stem
            predLabel = getSingleStemPred(img200Crop, model=stemUpClassifier if stem.isup else stemDownClassifier)
            stem.rhythm = predLabelConvert[predLabel]
            stem.isSingle = True
        xcenter = stem.start[0]
        if len(stem.alternativeBox)==0:
            stemX0 = xcenter - bar_height//3 #(bar_height*2//3 if stem.isup else bar_height//3)
            stemX1 = xcenter + bar_height//3 #(bar_height//3 if stem.isup else bar_height*2//3)
            stemY0,stemY1 = stem.getY0Y1()
        else:
            # x0,y0,x1,y1, x0 == x1
            x00, stemY0, xcenter2, stemY1 =stem.alternativeBox[0]
            assert x00 == xcenter2
            stemX0 = xcenter2 - bar_height//3
            stemX1 = xcenter2 + bar_height//3
        
        if stemY0 == stemY1:
            stemY1 = stemY0+1
        elif len(stem.alternativeBox)==0:
            # y0 is the top, smaller one
            stemY0 = stemY0 - beamMapImg[stemY0, -1, 2]
            stemY1 = stemY1 + beamMapImg[stemY1, -2, 2]
        cropPart = noteGroupStemMap[stemY0:stemY1, stemX0:stemX1]
        staffNumPart = mapStaffNum[stemY0:stemY1, stemX0:stemX1]
        staffNum = np.max(staffNumPart)
        # there's no overlapping group
        if np.max(cropPart) == 0:
            noteGroupStemMap[stemY0:stemY1, stemX0:stemX1] = len(noteGroupList)
            mapStaffNum[stemY0:stemY1, stemX0:stemX1] = staffNum
            noteGroupList.append(NoteGroup(stem))
        else:
            overlappingIndex = np.unique(cropPart).tolist()
            if 0 in overlappingIndex:
                overlappingIndex.remove(0)
            if len(overlappingIndex)==1:
                currGroupIdx = overlappingIndex[0]
                # xcenter-bar_height//2:xcenter+bar_height//2
                noteGroupStemMap[stemY0:stemY1, stemX0:stemX1] = currGroupIdx
                yy,xx = np.where(noteGroupStemMap == currGroupIdx)
                mapStaffNum[yy,xx] = np.max(mapStaffNum[yy,xx])
                noteGroupList[currGroupIdx] = mergeNoteGroup(noteGroupList[currGroupIdx], NoteGroup(stem), beamMapImg)
                # noteGroupList[currGroupIdx].addNoteStem(stem)
            else:
                currGroupIdx = overlappingIndex[0]
                noteGroupStemMap[stemY0:stemY1, stemX0:stemX1] = currGroupIdx
                for gIdx in overlappingIndex[1:]:
                    ys,xs = np.where(cropPart == gIdx)
                    noteGroupStemMap[ys,xs] = currGroupIdx
                    noteGroupList[currGroupIdx] = mergeNoteGroup(noteGroupList[currGroupIdx],noteGroupList[gIdx], beamMapImg)
                    noteGroupList[gIdx] = None
                yy,xx = np.where(noteGroupStemMap == currGroupIdx)
                mapStaffNum[yy,xx] = np.max(mapStaffNum[yy,xx])
                noteGroupList[currGroupIdx] = mergeNoteGroup(noteGroupList[currGroupIdx], NoteGroup(stem), beamMapImg)
    
    noteSizes = []
    for ng in noteGroupList:
        if ng is None:
            continue
        for stem in ng.noteStemList:
            imgrgb = cv2.rectangle(imgrgb, (stem.getX()-bar_height//3, stem.getTopCoord()[1]), (stem.getX()+bar_height//3, stem.getBottomCoord()[1]),
                                class_colors[stem.rhythm], 1, cv2.LINE_AA)
            x0,y0,x1,y1 = stem.noteBox
            noteSizes.append((x1-x0)*(y1-y0))
            imgrgb = cv2.rectangle(imgrgb, (x0,y0),(x1,y1), class_colors[stem.rhythm], 1, cv2.LINE_AA)
            if len(stem.alternativeBox) >0:
                alterBoxes = stem.alternativeBox
                for alterBox in alterBoxes:
                    imgrgb = cv2.rectangle(imgrgb, (alterBox[0], alterBox[1]),(alterBox[2],alterBox[3]),(120,150,255), 3, cv2.LINE_AA)
    kmeans = KMeans(n_clusters = 2, init = np.array([min(noteSizes), max(noteSizes)], dtype=np.uint16).reshape(-1, 1), n_init = 1)
    kmeans.fit(np.array(noteSizes,dtype=np.uint16).reshape(-1,1))
    
    if kmeans.cluster_centers_[1]/kmeans.cluster_centers_[0]>2:
        imgrgb2 = image.copy()
        currentCount = -1
        minSize = int(kmeans.cluster_centers_[1]//2)
        for ngId, ng in enumerate(noteGroupList):
            if ng is None:
                continue
            stmToRemove = []
            for stemId, stem in enumerate(ng.noteStemList):
                x0,y0,x1,y1 = stem.noteBox
                currentCount = currentCount+1
                # if kmeans.labels_[currentCount] == 0:
                #     break
                if (x1-x0)*(y1-y0)<minSize:
                    # TODO !!! potentially will filter out the big notes that are grouped to small ones
                    for stem in ng.noteStemList:
                        stem.isOrnament = True
                    stmToRemove.append(stemId)
                    continue
                imgrgb2 = cv2.rectangle(imgrgb2, (stem.getX()-bar_height//3, stem.getTopCoord()[1]), (stem.getX()+bar_height//3, stem.getBottomCoord()[1]),
                                    class_colors[stem.rhythm], 1, cv2.LINE_AA)
                
                imgrgb2 = cv2.rectangle(imgrgb2, (x0,y0),(x1,y1), class_colors[stem.rhythm], 1, cv2.LINE_AA)
                if len(stem.alternativeBox) >0:
                    alterBoxes = stem.alternativeBox
                    for alterBox in alterBoxes:
                        imgrgb2 = cv2.rectangle(imgrgb2, (alterBox[0], alterBox[1]),(alterBox[2],alterBox[3]),(120,150,255), 3, cv2.LINE_AA)
            stmToRemove.reverse()
            for rmId in stmToRemove:
                del ng.noteStemList[rmId]
            if len(ng.noteStemList) == 0:
                noteGroupList[ngId] = None
    else:
        imgrgb2 = imgrgb
    noteGroupMap = np.zeros((beamMapImg.shape[0], beamMapImg.shape[1]), dtype=np.uint16)
    ngImg = image.copy()

    colors = [(0,0,255),(0,255,0),(255,255,0), (0,120,230)]
    for idx,ng in enumerate(noteGroupList):
        if ng is not None:
            x0,y0,x1,y1 = ng.boundingBox
            noteGroupMap = cv2.rectangle(noteGroupMap, (x0,y0),(x1,y1), idx, -1)
            staffnum = np.max(mapStaffNum[y0:y1, x0:x1])
            mapStaffNum[y0:y1+1, x0:x1+1] = staffnum
            coloridx = -1 if staffnum ==0 else staffnum%3
            ngImg = cv2.rectangle(ngImg, (x0,y0),(x1,y1), colors[coloridx], 1, cv2.LINE_AA)
            for nb in ng.noteBoxes:
                xx0,yy0,xx1,yy1 = nb
                ngImg = cv2.rectangle(ngImg, (xx0,yy0),(xx1,yy1), colors[coloridx], 1, cv2.LINE_AA)
    beamMapImg = cv2.merge([mapBeamVal, middle, mapStaffNum]) 
    imwrite('ngImg.jpg',ngImg)
    imwrite('knnbeams.jpg',imgrgb)
    imwrite('knnbeams2.jpg',imgrgb2)
    global DEBUGIMG
    DEBUGIMG = imgrgb.copy()
    return noteGroupList,  beamMapImg, noteGroupStemMap, noteGroupMap, {'knnbeams':imgrgb,'ngImg': ngImg, 'knnbeams2': imgrgb2}

# savebeam is a black image with white on the place of beam and notes and clefs
def generateBeamStaffImg(savebeam:np.ndarray, staffObjList:List[Staff], barheight:int,stepSize=1):
    # beamThick = cv2.dilate(savebeam, np.ones((3,1), dtype=np.uint8), iterations=1)    
    mappingImgRgb = np.zeros((savebeam.shape[0], savebeam.shape[1], 3))
    savebeambw = cv2.cvtColor(savebeam, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(savebeambw.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    selectedContours = []
    bb,gg,rr = cv2.split(mappingImgRgb)
    staffYCenters = []
    for idx,staff in enumerate(staffObjList):
        if staff.left <= 4:
            staff.left = 5
        if staff.right >= savebeam.shape[1]-3:
            staff.right = savebeam.shape[1]-4
        staffYs = staff.ys
        oneY:float = staff.get_yOne_float()
        staffYCenter = staffYs[2]
        rr = cv2.rectangle(rr, (staff.left, staffYs[0]), (staff.right, staffYs[-1]), idx+1,-1)
        # -1: how far to the top [0]
        rr[staffYs[0]:min(staffYs[-1]+barheight, savebeam.shape[0]),-1] = np.array(range(0, min(staffYs[-1]+barheight, savebeam.shape[0])-staffYs[0]))
        rr[max(staffYs[0]-barheight, 0):staffYs[-1],-2] = np.array(range(staffYs[-1]-max(staffYs[0]-barheight, 0),0,-1))
        # line one's index will be 1~10:1, 2s will be 20~200:20
        indexRatio = (idx%2)*20+(1-idx%2)*1
        staffYCenter = staffYs[2]
        staffYCenters.append(staffYCenter)

        for i in range(1,13):
            if staffYCenter-i*oneY>0:
                rr[round(staffYCenter-i*oneY):round(staffYCenter-(i-1)*oneY),(idx+1)%4] = i+12
            else:
                break
        for i in range(1,13):
            if staffYCenter+i*oneY<rr.shape[0]:
                rr[round(staffYCenter+(i-1)*oneY):round(staffYCenter+i*oneY),(idx+1)%4] = 13-i
            else:
                break
    staffSeperationYs = [(staffYCenters[i]+staffYCenters[i-1])//2 for i in range(1,len(staffYCenters))]
    staffSeperationYs.insert(0,0)
    staffSeperationYs.append(savebeam.shape[0])
    # now it's the threshold for each line of staff
    for staffIdx in range(1,len(staffSeperationYs)):
        rr[staffSeperationYs[staffIdx-1]:staffSeperationYs[staffIdx], 4] = staffIdx
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w>barheight*1.5 and w<savebeam.shape[1]//3:
            area = cv2.contourArea(cnt)
            if area>w and not (w<2*barheight and h/w>0.5)and w>h:
                selectedContours.append(cnt)
    bb = cv2.drawContours(bb, selectedContours,-1, 255,thickness=-1)
    imwrite('beamOri.jpg',savebeambw)
    mappingImgRgb = cv2.merge([bb,gg,rr])
    imwrite('beamOricontours.jpg',mappingImgRgb)
    bbUint8 = bb.astype(np.uint8)
    kernel = np.ones((1,barheight//3), dtype=np.uint8)
    bbUint8 = cv2.dilate(bbUint8, kernel, iterations=1)
    mappingImgUint8= cv2.merge([bbUint8,gg.astype(np.uint8),rr.astype(np.uint8)])
    print('preparing for beam image')
    for xIdx in range(0,savebeam.shape[1], stepSize*2):
        # no white in this place
        if sum(mappingImgUint8[:,xIdx,0]) == 0:
            continue
        startingY = np.where(mappingImgUint8[:,xIdx,0]==255)[0].tolist()
        # the ending of the bottommost one
        startingY.append(min(mappingImgUint8.shape[0],startingY[-1]+barheight*5))
        for i in range(0,len(startingY)-1):
            yStartVal = 253
            for yIdx in range((startingY[i]+stepSize)//(2*stepSize)*(2*stepSize)+stepSize, startingY[i+1],stepSize*2):
                if yStartVal<=0:
                    break
                mappingImgUint8[yIdx,xIdx,0]=yStartVal
                yStartVal-=stepSize*2
        # the ending of the topmost one
        startingY.pop()
        startingY.insert(0, max(startingY[0]-barheight*5, 0))
        for i in range(len(startingY)-1, 0,-1):
            yStartVal = 254
            for yIdx in range((startingY[i]-1)//(2*stepSize)*(2*stepSize), startingY[i-1],-stepSize*2):
                if yStartVal<=0:
                    break
                mappingImgUint8[yIdx,xIdx,0]=yStartVal
                yStartVal-=stepSize*2
    onlyBImg = mappingImgUint8.copy()
    onlyBImg[:,:,1] = onlyBImg[:,:,0]
    onlyBImg[:,:,2] = onlyBImg[:,:,0]
    imwrite('onlyBImg.jpg',onlyBImg)
    saveImages = {}
    saveImages['gradImg']=onlyBImg
    saveImages['beamContour']=mappingImgRgb
    return mappingImgUint8, saveImages

def getBeamImage(dataDict, bar_height, staffObjList:List[Staff]):
    stems_rests = dataDict['stems_rests']
    clefs_keys = dataDict['clefs_keys']
    notehead = dataDict['notehead']
    symbols = dataDict['symbols']
    image:np.ndarray = dataDict['image']

    _, img128 = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((bar_height//3, bar_height//3),dtype=np.uint8)
    beam = cv2.erode(img128, kernel, iterations = 1)
    imwrite('note_beam1.jpg',beam)

    for ll in range(3):
        haslinecount = 0
        totalCount = 0
        for staff in staffObjList:
            for b in range(len(staff.ys)):
                curr = sum(np.max(beam[max(staff.ys[b]-bar_height//3,0):min(staff.ys[b]+bar_height//3, beam.shape[0]), staff.left:staff.right,1],0)==255)
                if curr>(staff.right-staff.left)//3:
                    haslinecount+=1
                totalCount+=1
        if haslinecount/totalCount>0.6:
            kernel = np.ones((bar_height//4, bar_height//4),dtype=np.uint8)
            beam = cv2.erode(beam, kernel, iterations = 1)
        else:
            break
    imwrite('note_beam1.5.jpg',beam)
    xs,ys = np.where(clefs_keys>0)
    beam[xs,ys] = (0,0,0)
    beamWithoutStemRest = beam.copy()
    imwrite('note_beam2.jpg',beamWithoutStemRest)
    # beamret is beamNoClefKey
    xs,ys = np.where(stems_rests>0)
    beam[xs,ys] = (255,255,255)
    imwrite('note_beam3.jpg', beam)
    # beam is beam with stem_rests

    kernel_tall = np.ones((1,bar_height*2), np.uint8)
    beam_nostem = cv2.erode(beam, kernel_tall)
    beam_nostem = cv2.dilate(beam_nostem, kernel_tall)
    imwrite('note_beam4.jpg', beam_nostem)
    beam_nostem_nostaff = beam_nostem.copy()
    savebeambw = cv2.cvtColor(beam_nostem, cv2.COLOR_BGR2GRAY)
    toDel = np.where(np.sum(savebeambw>127,axis=1)>savebeambw.shape[1]*0.4)[0] # VARIABLE threshold to delete
    for d in toDel:
        beam_nostem_nostaff[d,:] = 0
    imwrite('note_beam5.jpg', beam_nostem_nostaff)
    # beam nostem is beam with stemRest + erode (also has notes)
    return beam, beamWithoutStemRest, beam_nostem_nostaff

def filter_boxes_on_beam(notehead_boxes, beamMapImg, areamin):
    filtered_Boxes = []
    for box in notehead_boxes:
        x0,y0,x1,y1 = box
        if np.max(beamMapImg[y0:y1, x0:x1, 0])==255:
            w = x1-x0
            h = y1-y0
            if w*h<areamin:
                continue
        filtered_Boxes.append(box)
    return filtered_Boxes

# in here the beam is beamNoStem, beamMapImg[ , ,0]=255: it's possibly a stem
def getInitialNoteheadBoxList(dataDict, beam_nostem, beamMapImg:np.ndarray, bar_height):
    notehead = dataDict['notehead']
    image:np.ndarray = dataDict['image']
    xs,ys = np.where(cv2.cvtColor(beam_nostem, cv2.COLOR_BGR2GRAY)>0)
    notehead_mod = notehead.copy()*255
    notehead_mod[xs,ys] = 0

    imwrite('notehead_ori.jpg', notehead_mod)
    dilation_ratio1 = 0.33
    dilation_ratio2 = 0.4
    # dilate->erode twice to clear image + get clear noteheads
    size1 = int(round(bar_height*dilation_ratio1))
    note_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size1, size1))
    notehead_mod = cv2.erode(cv2.dilate(notehead_mod.astype(np.uint8), note_kernel), note_kernel)
    size2 = int(round(bar_height*dilation_ratio2))

    note_kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size2,size2))
    notehead_mod2 = cv2.erode(notehead_mod, note_kernel2)
    note_kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size2+1, size2+1))
    notehead_mod2 = cv2.dilate(notehead_mod2, note_kernel3)

    # both b&w image, white is where there's stuff
    imwrite('notehead_mod.jpg',notehead_mod)
    imwrite('notehead_mod2.jpg',notehead_mod2)
    # notehead mod is basically the cleaned beams

    # get the boxes, if the boxes is in 
    notehead_boxes_all = get_bbox(notehead_mod2)
    notehead_boxes_big, notehead_mod3= filter_small_bbox_area(notehead_boxes_all, notehead_mod2,xmin = bar_height*0.5,ymin = bar_height*0.2, areamin = bar_height*bar_height*0.4)
    imwrite('notehead_mod3.jpg',notehead_mod3)
    notehead_boxes = split_connected_notes(notehead_boxes_big, bar_height,notehead_mod2)
    notehead_boxes_filtered = filter_boxes_on_beam(notehead_boxes, beamMapImg,areamin = bar_height*bar_height*0.8)
    return notehead_boxes_filtered, notehead_mod2

# noteheadInitial : b&w 2d image, 255: notehead, 
# beamMapImg: gradient image, the 1st dimension is what we care about
def getStemList(dataDict, notehead_boxes:List[Tuple[int,int,int,int]], beamWithStemRest):
    image:np.ndarray = dataDict['image']

    drawimg = image.copy()
    # 255 is where there's stuff, 0 is empty
    _, img200 = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY_INV)
    imgh = image.shape[0]
    notehead_rectangle = np.zeros((image.shape[0],image.shape[1]))
    stem_init_lst:List[Stem]=[]
    # beambw is suppose to be white where there's beam
    beambw = cv2.cvtColor(beamWithStemRest, cv2.COLOR_BGR2GRAY)
    imwrite('beambw.jpg',beambw)
    # get the stem's directions
    for idx,boxOriginal in enumerate(notehead_boxes):
        x0,y0,x1,y1 = boxOriginal
        xleft = (x0+x1*2)//3
        xright = (x0*2+x1)//3

        leftRange = np.where(img200[y0:y1,xleft]==255)[0].tolist()
        rightRange = np.where(img200[y0:y1, xright] ==255)[0].tolist()
        if len(leftRange)*len(rightRange) != 0:
            y1 = y0+max(max(leftRange),max(rightRange))
            y0 += min(min(leftRange), min(rightRange))
            if y1<=y0:
                x0,y0,x1,y1 = boxOriginal
        box = (x0,y0,x1,y1)
        notehead_rectangle[y0:y1,x0:x1] = 1
        length = bar_height*3
        width = (x1-x0)//3
        leftXrange = range(x0-width//2,x0+width+1)
        rightXrange = range(x1-width, x1+width//2+1)
        topYrange = range(max(y0-length, 0),y0)
        bottomYrange = range(y1,min(y1+length, imgh))
        left_area = beambw[bottomYrange.start:bottomYrange.stop,
                         leftXrange.start: leftXrange.stop]    
        left_centerX = np.argmax(np.sum(left_area,0))
        right_area = beambw[topYrange.start:topYrange.stop,
                         rightXrange.start:rightXrange.stop]
        right_centerX = np.argmax(np.sum(right_area,0))
        # has to be more strict not too far to the left
        leftup_area = beambw[topYrange.start:topYrange.stop,
                         leftXrange.start: leftXrange.stop]
        leftup_centerX = np.argmax(np.sum(leftup_area,0))
        rightdn_area = beambw[bottomYrange.start:bottomYrange.stop,
                              rightXrange.start:rightXrange.stop]
        rightdn_centerX = np.argmax(np.sum(rightdn_area,0))
        isup = True
        hasStem = True
        alterBox = []
        leftRightRatio = np.sum(left_area[:,left_centerX])/np.sum(right_area[:, right_centerX])
        if leftRightRatio>1:
            # stem is in left bottom
            maxRange = [leftXrange, bottomYrange]
            isup = False
            if leftRightRatio<1.25:
                alterBox.append([rightXrange.start+right_centerX,topYrange.start,rightXrange.start+right_centerX,topYrange.stop])
        else:
            # it's either close enough or in right bottom
            maxRange = [rightXrange, topYrange]
            # they are close enough -> alterbox will always record the left bottom one
            if leftRightRatio>0.8:
                alterBox.append([leftXrange.start+left_centerX,bottomYrange.start,leftXrange.start+left_centerX,bottomYrange.stop])
        maxArea = max(np.sum(left_area[:,left_centerX]),np.sum(right_area[:, right_centerX]))
        maxAlterArea = max(np.sum(leftup_area[:,leftup_centerX]),np.sum(rightdn_area[:,rightdn_centerX]))
        if maxAlterArea>255*length//2 and maxAlterArea/maxArea>1.5:
            if np.sum(leftup_area[:,leftup_centerX])>np.sum(rightdn_area[:,rightdn_centerX]):
                alterBox.append([leftXrange.start+leftup_centerX, topYrange.start, leftXrange.start+leftup_centerX, topYrange.stop])
            else:
                alterBox.append([rightXrange.start+rightdn_centerX, bottomYrange.start, rightXrange.start+rightdn_centerX, bottomYrange.stop])
        if len(alterBox)==0 and maxArea < 255*length//3:
            hasStem = False
            # hasStem is a flag for later grouping notes
        xrange, yrange = maxRange
        crop = img200[yrange.start:yrange.stop, xrange.start:xrange.stop]
        a = (np.where(np.sum(crop,0)==np.max(np.sum(crop,0)))[0]).tolist()
        chosenx = a[len(a)//2]
        x_y = (xrange.start+chosenx, y0 if isup else y1)
        stem_init_lst.append(Stem(x_y, isup,boxOriginal,box, hasStem=hasStem, alterbox=alterBox))
        if len(alterBox)==0 and hasStem:
            drawimg = cv2.rectangle(drawimg, (xrange.start+chosenx-1, yrange.start),(xrange.start+chosenx+1,yrange.stop),(255,0,120), 1, cv2.LINE_AA)
        elif hasStem:
            drawimg = cv2.putText(drawimg, str(idx), (x0,y0),cv2.FONT_HERSHEY_SIMPLEX,  1, (120,0,120), 2, cv2.LINE_AA)
            drawimg = cv2.rectangle(drawimg, (xrange.start+chosenx-1, yrange.start),(xrange.start+chosenx+1,yrange.stop),(255,255,0), 2, cv2.LINE_AA)
            for ab in alterBox:
                drawimg = cv2.rectangle(drawimg, (ab[0], ab[1]),(ab[2],ab[3]),(0,255,0), 3, cv2.LINE_AA)
        drawimg = cv2.rectangle(drawimg, (x0,y0),(x1,y1),(0,0,255), 2, cv2.LINE_AA)
    imwrite('notehead_drawimg.jpg',drawimg)
    return stem_init_lst, image, {'notehead_drawimg':drawimg}

def assignStemLength(stem_init_lst:List[Stem], image, beamMapImg:np.ndarray, stepsize: int, bar_height):
    # get the stem's length
    img200 =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,img200bw = cv2.threshold(img200, 200, 255, cv2.THRESH_BINARY_INV)
    # img200 is black and white, white is where there's stem
    imwrite('img200inv.jpg', img200bw)
    # wid = 1
    topthres = bar_height//2
    ending_black_height = 3
    # make it whiter for better visualization
    drawimg2 = np.ones_like(img200bw)*255 - img200bw.copy()//4
    drawimg2 = cv2.cvtColor(drawimg2, cv2.COLOR_GRAY2BGR)
    bigStep = stepSize*2
    for idx,coordStart in enumerate(stem_init_lst):
        x,yOri = coordStart.start
        wid = (coordStart.noteBox[2]-coordStart.noteBox[0])//3
        yend = yOri
        sign = 1 if coordStart.isup else -1
        # is in staff +- 3.5 barheight, no alternative box, has stem
        # +beamMapImg[max(yOri-int(2.5*bar_height)*sign,0),x,2]
        isTypical:bool = (len(coordStart.alternativeBox) == 0) and coordStart.hasStem and min([abs(posIdx-11.5) for posIdx in beamMapImg[yOri,0:4,2]])<5

        # if the beamMap is less than 4*barheight away and the beam itself is somewhere in a staffArea, assign it
        # elif it's not too far away (2*4*barheight), see if the area has at least 1/2 blacks
        # else it's gonna go through the clssification model to see if it's a single quarter/eighth/sixteenth
        xLeft = x//bigStep*bigStep
        xRight = min(beamMapImg.shape[1]-1,x//bigStep*bigStep+bigStep)
        staffLineNumber:int = beamMapImg[yOri, 4, 2]
        barPositionI: int = beamMapImg[yOri, staffLineNumber%4, 2]
        stemUpLongest:int = 0 if (barPositionI<8 or barPositionI>17) else 17-barPositionI
        stemDownLongest:int = 0 if (barPositionI<8 or barPositionI>17) else barPositionI-8
        if coordStart.isup:
            longestNormal = stemUpLongest
            # go how far is considered normal and ok
            y = yOri//bigStep*bigStep-stepsize
            leftUpValue = beamMapImg[y,xLeft, 0]
            rightUpValue = beamMapImg[y,xRight, 0]
            if leftUpValue>rightUpValue:
                maxMapValue = leftUpValue
            else:
                maxMapValue = rightUpValue
            assert maxMapValue%2==1 or maxMapValue == 0
            if maxMapValue != 0:
                if isTypical and (253-maxMapValue)+bigStep<=(longestNormal)*bar_height:
                    if maxMapValue < 255:
                        yend = y-(253-maxMapValue)
                        assert max(beamMapImg[yend, xLeft,0],beamMapImg[yend, xRight,0]) >= 253
                        while max(beamMapImg[yend, xLeft,0],beamMapImg[yend, xRight,0]) >= 253-bar_height//2:
                            yend -= bigStep
                        while max(beamMapImg[yend, xLeft,0],beamMapImg[yend, xRight,0]) == 255:
                            yend -= 1
                        # the end of the stem doesn't have enough "stem (white)" (the image is inversed)
                        if (253-maxMapValue)+bigStep>(longestNormal-3)*bar_height and (sum(np.max(img200bw[yend:y, x-wid:x+wid+1],1))<(y-yend)*255*0.8 or sum(np.max(img200bw[yend:(y+yend)//2, x-wid:x+wid+1],1))<(y-yend)*255*0.4):
                            yend = yOri
            if yend==yOri:
                if isTypical and coordStart.hasStem and len(coordStart.alternativeBox)==0:
                    yend -= min(3,longestNormal)*bar_height
                # while there is still things in the inversed bw original image
                while sum(np.max(img200bw[yend-ending_black_height:yend, x-wid:x+wid+1],1))!=0:
                    yend -= 1
            else:
                while sum(np.max(img200bw[yend-ending_black_height:yend, x-1:x+2],1))!=0: # and max([(10-beamMapImg[yend,x,1]%11)%10,(10-beamMapImg[yend,x,1]//20)%10]): 
                    yend -= 1
        else:
            longestNormal = stemDownLongest
            y = min(yOri//bigStep*bigStep+bigStep,beamMapImg.shape[0]-1)
            leftUpValue = beamMapImg[y,xLeft, 0]
            rightUpValue = beamMapImg[y,xRight, 0]
            if leftUpValue>rightUpValue:
                maxMapValue = leftUpValue
            else:
                maxMapValue = rightUpValue
                xRight = x//bigStep*bigStep+bigStep
            assert maxMapValue%2==0 or maxMapValue == 255
            if isTypical and (254-maxMapValue)+bigStep<=longestNormal*bar_height:
                if maxMapValue < 255:
                    yend = y+(254-maxMapValue)
                    assert max(beamMapImg[yend, xLeft,0],beamMapImg[yend, xRight,0]) >=254
                    while max(beamMapImg[yend, xLeft,0],beamMapImg[yend, xRight,0]) >= 254-bar_height//2 and (yend-yOri)<=(longestNormal)*bar_height:
                        yend += bigStep
                    while max(beamMapImg[yend, xLeft,0],beamMapImg[yend, xRight,0]) == 255 and (yend-yOri)<=(longestNormal)*bar_height:
                        yend += 1
                    # the end of the stem doesn't have enough "stem (white)" (the image is inversed)
                    if (254-maxMapValue)+bigStep>(longestNormal-3)*bar_height and (sum(np.max(img200bw[y:yend, x-wid:x+wid+1],1))<(yend-y)*128 or sum(np.max(img200bw[(y+yend)//2:yend, x-wid:x+wid+1],1))<(yend-y)*64):
                        yend = yOri
            if yend==yOri:
                if isTypical and coordStart.hasStem and len(coordStart.alternativeBox)==0:
                    yend += min(3,longestNormal)*bar_height
                # while there is still things in the inversed bw original image
                while sum(np.max(img200bw[yend:yend+ending_black_height, x-wid:x+wid+1],1))!=0 and (yend-yOri)<=(longestNormal)*bar_height: 
                    yend += 1
            else:
                while sum(np.max(img200bw[yend:yend+ending_black_height, x-1:x+2],1))!=0: # and min([beamMapImg[yend,x,1]%11,beamMapImg[yend,x,1]//20])<2:
                    yend += 1
        if abs(yend-y)<bar_height and not coordStart.hasStem:
            yend = yOri
        coordStart.setYlen(yend-yOri)
        assert coordStart.end is not None
        drawimg2 = cv2.rectangle(drawimg2, coordStart.getTopCoord(), coordStart.getBottomCoord(),(255,0,255), 1, cv2.LINE_AA)
        if not isTypical:
            drawimg2 = cv2.putText(drawimg2, str(idx), coordStart.end, cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1, cv2.LINE_AA)
        else:
            drawimg2 = cv2.putText(drawimg2, str(idx), coordStart.end, cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1, cv2.LINE_AA)
    imwrite('notestem_drawimg.jpg',drawimg2)
    return stem_init_lst, {'notestem_drawimg': drawimg2}


def maskImage(noteheadBoxesInitList:List[Tuple[int,int,int,int]], noteBwImageOrigin: np.ndarray):
    maskedNoteBw = np.zeros_like(noteBwImageOrigin)
    for box in noteheadBoxesInitList:
        x0,y0,x1,y1 = box
        maskedNoteBw[y0:y1, x0:x1] = noteBwImageOrigin[y0:y1, x0:x1]
    imwrite('maskedNoteBw.jpg',maskedNoteBw)
    return maskedNoteBw

# beamNoClefKeyWithStemRest: 3d, maskedNoteBwImage: 1d, beamMapImg: 3d, we only look at the b part [:,:,0]
def removeNotes(maskedNoteBwImage:np.ndarray, beamNoClefKeyWithStemRest:np.ndarray, beamMapImg:np.ndarray):
    beambw = beamMapImg[:,:,0]
    xs,ys = np.where(maskedNoteBwImage>0) # where there's note but no beam
    beamNoNotes = beamNoClefKeyWithStemRest.copy()
    beamNoNotes[xs,ys] = (0,0,0)
    imwrite('beamNoNotes1.jpg',beamNoNotes)
    xs,ys = np.where(beambw==255)
    beamNoNotes[xs,ys] = (255,255,255)
    imwrite('beamNoNotes2.jpg',beamNoNotes)

    return beamNoNotes

def mergeVerticalNoteGroups(staffObjList:List[Staff],noteGroupList:List[NoteGroup], noteGroupMap:np.ndarray,beamMapImg:np.ndarray, image:np.ndarray,
                            restMap, restList:List[Rest]):
    for staff in staffObjList:
        barh = staff.get_yOne()
        oneStep = barh//3
        y0 = max(staff.ys[0]-int(barh*2.5),0)
        y1 = min(staff.ys[-1]+int(barh*2.5), noteGroupMap.shape[0])
        tempRestMap = restMap.copy()
        for x in range(staff.left, staff.right-oneStep, oneStep):
            indexes = set(np.unique(noteGroupMap[y0:y1, x]))
            indexesNext = set(np.unique(noteGroupMap[y0:y1, x+oneStep]))
            rIndex = set(np.unique(tempRestMap[y0:y1, x]))
            rIndexNext = set(np.unique(tempRestMap[y0:y1, x+oneStep]))
            line = set(np.unique(beamMapImg[y0:y1, x:x+oneStep,2]))
            indexes -= {0}
            indexesNext-={0}
            line-={0}
            rIndex-={0}
            rIndexNext-={0}
            if len(line)>1:
                continue
            if len(indexes) == 0:
                continue
            # the overlapping x width is at least oneStep width
            noteGroupId = list(indexes)[0]
            lineList = list(line)
            idxList = list(indexes)

            if len(indexes)>1 and indexes==indexesNext: # two noteGroup overlap
                noteGroupId = idxList.pop()
                for gid in idxList:
                    ys,xs = np.where(noteGroupMap == gid)
                    noteGroupMap[ys,xs] = noteGroupId
                    beamMapImg[ys, xs, 2] = lineList[0]
                    noteGroupList[noteGroupId]=mergeNoteGroup(noteGroupList[noteGroupId], noteGroupList[gid], beamMapImg)
                    noteGroupList[gid] = None
            if len(rIndex)>0 and rIndex == rIndexNext:
                ridLst = list(rIndex)
                for rid in ridLst:
                    ys,xs = np.where(tempRestMap == rid)
                    if restList[rid].rhythm == -1:
                        continue
                    else:
                        tempRestMap[ys,xs] = 0
                        beamMapImg[ys,xs,2] = lineList[0]
                        noteGroupList[noteGroupId].addRest(restList[rid])
                        restList[rid].setNgId(noteGroupId)                
    ngImg = image.copy()
    colors = [(0,0,255),(0,255,0),(255,255,0), (0,120,230)]
    for ng in noteGroupList:
        if ng is not None:
            x0,y0,x1,y1 = ng.boundingBox
            ngImg = cv2.rectangle(ngImg, (x0,y0),(x1,y1), colors[(np.max(beamMapImg[y0:y1, x0:x1, 2])%3 if np.max(beamMapImg[y0:y1, x0:x1, 2])>0 else 3)], 2, cv2.LINE_AA)
    imwrite('vertically_merged.jpg', ngImg)
    return noteGroupList, noteGroupMap, beamMapImg, {'vertically_merged':ngImg}

def getbbox(clefs_keys: np.ndarray, bar_height:int)-> List[Tuple[int,int,int,int]]:
    contours, _ = cv2.findContours(clefs_keys.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w>3 and h>bar_height*1.6 and h>w:
            box = (x, y, x+w, y+h)
            bboxes.append(box)
    return bboxes


def getBestRange(avg_color_lst: list, desire_length:int):
    if desire_length>=len(avg_color_lst)-1:
        return 0, len(avg_color_lst)-1
    most_white_value = 0
    most_white_ending_idx = 0
    for i in range(desire_length,len(avg_color_lst)-1):
        currSum = sum(avg_color_lst[i-desire_length:i])
        if currSum>most_white_value:
            most_white_value=currSum
            most_white_ending_idx = i
    return most_white_ending_idx-desire_length, most_white_ending_idx
def sfnClefToDataType(bbox:Tuple[int,int,int,int], predictionId: int) -> Union[Accidentals, Clef]:
    # 'BassF', 'ViolaC', 'flat', 'natural', 'noClass', 'sharp', 'trebleG'
    isClef = [True, True, False, False, False, False, True]
    mapId = [-1, 0, -1, 0, -3, 1, 1]
    # we should never try to assign a noClass
    assert predictionId != 4
    if isClef[predictionId]:
        return Clef(bbox, mapId[predictionId])
    else:
        return Accidentals(bbox, mapId[predictionId])
    
    
def symbol_classification(dataDict: dict, model:Sfn_Clef_classifier, bar_height:int):
    # output_dir will be images/tch/tch SYMBOL
    clefs_keys_ori:np.ndarray = dataDict['clefs_keys']
    clefs_keys = (clefs_keys_ori*255).astype(np.uint8)
    clefs_keys_expand = cv2.dilate(clefs_keys, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)))
    image:np.ndarray = dataDict['image']
    round1_sfn_img = image.copy()
    ori_bboxes = getbbox(clefs_keys_expand, bar_height)
    bboxes = merge_nearby_bbox(ori_bboxes, bar_height*3)
    class_names = ['BassF', 'ViolaC', 'flat', 'natural', 'noClass', 'sharp', 'trebleG','wide']  # Replace with actual class names
    class_symbols = ['Bass','Viola','b','n','x','#','Treble','Wide']
    class_colors = [(255,0,0),(0,255,0),(255,255,0),(255,0,125),(0,120,255),(255,0,255),(0,0,255),(101,134,168)]
    for j in range(0,len(class_colors)):
        round1_sfn_img = cv2.putText(round1_sfn_img,class_names[j], (30, 30+j*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_colors[j], 2, cv2.LINE_AA)
    noClass_bbox = []
    sfn_height_lst = []
    sfn_width_lst = []
    # the 0th is a stub for better alignment
    returnSfnList:List[Union[Accidentals,Clef, None]] = [None]
    for b in bboxes:
        if (b[3]-b[1])<(b[2]-b[0]): # height<width: wide
            crop_img = clefs_keys[b[1]:b[3], b[0]:b[2]]
            predict_idx = model.predictWide(crop_img)
            if predict_idx ==4:
                predict_idx = 7
        else:
            crop_img = clefs_keys[b[1]:b[3], b[0]:b[2]]
            predict_idx = model.predict(crop_img)
            if predict_idx == 0 and b[3]-b[1]<bar_height*2: # if it's Bass but really small -> likely it's flat instead
                predict_idx = 2
            if predict_idx == 2 or predict_idx==5 or predict_idx==3:
                # this is flat | sharp | natural
                sfn_height_lst.append(b[3]-b[1])
                sfn_width_lst.append(b[2]-b[0])
        round1_sfn_img = cv2.rectangle(round1_sfn_img, (b[0],b[1]),(b[2],b[3]), class_colors[predict_idx],2,cv2.LINE_AA)
        round1_sfn_img = cv2.putText(round1_sfn_img, class_symbols[predict_idx], (b[0],b[3]+10),cv2.FONT_HERSHEY_SIMPLEX,  1, class_colors[predict_idx], 1, cv2.LINE_AA)
        if predict_idx == 4 or predict_idx==7:
            noClass_bbox.append(b)
        else:
            returnSfnList.append(sfnClefToDataType(b, predict_idx))
    noClass_bbox2 = []
    round2_sfn_img = round1_sfn_img.copy()
    sfn_median_height = bar_height*2
    sfn_median_width = bar_height
    if len(sfn_height_lst)>0:
        sfn_median_height = int(np.median(sfn_height_lst))
        sfn_median_width = int(np.median(sfn_width_lst))
    
    for box1 in noClass_bbox:
        x0 = box1[0]
        y0 = box1[1]
        clef_key_crop = clefs_keys[box1[1]:box1[3], box1[0]:box1[2]]
        smallbbox = getbbox(clef_key_crop, bar_height)
        if len(smallbbox)<=1:
            if len(smallbbox)==1 and ((box1[2]-box1[0])>sfn_median_width*1.5 or (box1[3]-box1[1])>sfn_median_height*1.5):
                noClass_bbox2.append(box1)
            predict_idx = 4
            round2_sfn_img = cv2.rectangle(round2_sfn_img, (box1[0],box1[1]),(box1[2],box1[3]), class_colors[predict_idx],2,cv2.LINE_AA)
            round2_sfn_img = cv2.putText(round2_sfn_img, class_symbols[predict_idx], (box1[0],box1[3]+10),cv2.FONT_HERSHEY_SIMPLEX,  1, class_colors[predict_idx], 1, cv2.LINE_AA)
            continue
        for bb in smallbbox:
            b = [x0+bb[0],y0+bb[1],x0+bb[2],y0+bb[3]]
            small_crop = clefs_keys[b[1]:b[3], b[0]:b[2]]
            predict_idx = model.predict(small_crop)
            round2_sfn_img = cv2.rectangle(round2_sfn_img, (b[0],b[1]),(b[2],b[3]), class_colors[predict_idx],2,cv2.LINE_AA)
            round2_sfn_img = cv2.putText(round2_sfn_img, class_symbols[predict_idx], (b[0],b[3]+10),cv2.FONT_HERSHEY_SIMPLEX,  1, class_colors[predict_idx], 1, cv2.LINE_AA)

            if predict_idx == 4:
                if (small_crop.shape[0]>sfn_median_height*1.5 and small_crop.shape[1]>sfn_median_width*0.8) or (small_crop.shape[0]>sfn_median_height*0.8 and small_crop.shape[1]>sfn_median_width*1.5):
                    noClass_bbox2.append(b)
                else:
                    continue
            else:
                returnSfnList.append(sfnClefToDataType(b, predict_idx))
    for box1 in noClass_bbox2:
        x0 = box1[0]
        y0 = box1[1]
        w = box1[2]-x0
        h = box1[3]-y0
        seg_img = clefs_keys[y0:y0+h, x0:x0+w]
        width_ratio =w/sfn_median_width
        height_ratio = h/sfn_median_height
        num_symbol = round(max(width_ratio, height_ratio))
        if num_symbol<2:
            continue
        if width_ratio>height_ratio:
            left_center = sfn_median_width//2
            right_center = w-sfn_median_width//2
            avg_width:float = (right_center-left_center)/(num_symbol-1)
            for i in range(num_symbol):
                img_left = int(i*avg_width)
                crop_img = seg_img[:,img_left:min(img_left+sfn_median_width,seg_img.shape[1])]
                vertical_avg = np.mean(crop_img,1)
                front,back = getBestRange(vertical_avg, sfn_median_height)
                # b: x0, y0, x1, y1 relative to the croped image
                b = [max(img_left,0),max(front,0),
                     min(img_left+sfn_median_width,seg_img.shape[1]), min(back,seg_img.shape[0])]
                crop_crop_img = seg_img[b[1]:b[3], b[0]:b[2]]
                # predict_idx = model.predict(crop_crop_img)
                if sum(np.max(crop_crop_img,1))<=0:
                    continue
                if sum(np.max(crop_crop_img,1))/255/crop_crop_img.shape[0]<=0.6:
                    continue
                predict_lst = model.get_prediction_vector(crop_crop_img)
                pred_filter = [x for x in predict_lst if x not in [0,1,4,6,7]]
                predict_idx = pred_filter[0]
                returnSfnList.append(sfnClefToDataType((x0+b[0], y0+b[1], x0+b[2],y0+b[3]), predict_idx))
        else:
            seg_img_to_delete = seg_img.copy()
            vertical_boxes = []
            pred_idx = []

            b = [0,0,w,sfn_median_height] #x0, y0, x1, y1
            hor_avg_top = np.mean(seg_img[b[1]:b[3],:],0)
            b[0],b[2] = getBestRange(hor_avg_top, sfn_median_width)
            seg_img_to_delete[b[1]:b[3],b[0]:b[2]] = 0
            vertical_boxes.append(b)

            b = [0,h-sfn_median_height-1,w,h-1]
            hor_avg_bot = np.mean(seg_img[b[1]:b[3],:],0)
            b[0],b[2] = getBestRange(hor_avg_bot, sfn_median_width)
            seg_img_to_delete[b[1]:b[3],b[0]:b[2]] = 0
            vertical_boxes.append(b)

            if h>sfn_median_height*2:
                b = [0,0,0,0]
                # middle_del = seg_img_to_delete[sfn_median_height//2:h-sfn_median_height//2,:]
                ver_avg_mid = np.mean(seg_img_to_delete,1)
                b[1],b[3] = getBestRange(ver_avg_mid, sfn_median_height)
                expanded_left_right = [min(vertical_boxes[0][0],vertical_boxes[1][0]),
                                       max(vertical_boxes[0][2],vertical_boxes[1][2])]
                if expanded_left_right[0]>=w-expanded_left_right[1]-sfn_median_width//5:
                    b[2] = sfn_median_width
                else:
                    b[0] = w-sfn_median_width-1
                    b[2] = w-1
                vertical_boxes.append(b)
                seg_img_bgr = cv2.cvtColor(seg_img,cv2.COLOR_GRAY2BGR)
                seg_img_bgr = cv2.rectangle(seg_img_bgr, (b[0],b[1]),(b[2],b[3]),(0,255,0),2,cv2.LINE_AA)

            for idx, b in enumerate(vertical_boxes):
                predict_lst = model.get_prediction_vector(seg_img[b[1]:b[3],b[0]:b[2]])
                pred_filter = [x for x in predict_lst if x not in [0,1,4,6,7]]
                predict_idx = pred_filter[0]
                if idx == 2 and predict_lst[0] == 4:
                    continue
                returnSfnList.append(sfnClefToDataType((x0+b[0], y0+b[1], x0+b[2],y0+b[3]), predict_idx))
    sfnClefImg = image.copy()
    class_colors2 = [(255,255,0),(255,0,125),(255,0,255),(255,0,0),(0,255,0),(0,0,255)]
    class_names2 = ['flat', 'natural', 'sharp','BassF', 'ViolaC', 'trebleG']
    for j in range(0,len(class_colors2)):
        sfnClefImg = cv2.putText(sfnClefImg,class_names2[j], (30, 30+j*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_colors2[j], 2, cv2.LINE_AA)
    # class_names = ['BassF', 'ViolaC', 'flat', 'natural', 'noClass', 'sharp', 'trebleG','wide']
    
    sfnClefMap = np.zeros((sfnClefImg.shape[0], sfnClefImg.shape[1]), dtype=np.uint16)
    for idx,clefAccidentals in enumerate(returnSfnList):
        if clefAccidentals is not None:
            x0,y0,x1,y1 = clefAccidentals.getBbox()
            # 0 if it's accidentals
            predict_idx = clefAccidentals.getType()*3+clefAccidentals.getValue()+1
            sfnClefImg = cv2.rectangle(sfnClefImg, (x0,y0),(x1,y1), class_colors2[predict_idx],2,cv2.LINE_AA)
            sfnClefMap = cv2.rectangle(sfnClefMap, (x0,y0),(x1,y1), idx, -1)

    imwrite('round1Sfn.jpg',round1_sfn_img)
    imwrite('round2Sfn.jpg',round2_sfn_img)
    imwrite('round3Sfn.jpg', sfnClefImg)
    
    return returnSfnList, sfnClefMap, {'sfnClef':sfnClefImg}

# filter out the sfn that is overlapping with noteGroupList
def filterSfnClefModifyBeamMap(sfnClefList:List[Union[Accidentals, Clef, None]], 
                               noteGroupMap: np.ndarray, 
                               beamMapImg:np.ndarray):
    sfnClefMap = np.zeros((beamMapImg.shape[0], beamMapImg.shape[1]), dtype=np.uint16)
    bb,gg,rr = cv2.split(beamMapImg)
    assert len(np.unique(gg)) <=2
    for idx, sfnClef in enumerate(sfnClefList):
        if sfnClef is not None:
            x0,y0,x1,y1 = sfnClef.getBbox()
            if np.sum(noteGroupMap[y0:y1,x0:x1]>0)/((x1-x0)*(y1-y0))>0.5:
                sfnClefList[idx] = None
            else:
                sfnClefMap[y0:y1,x0:x1] = idx
                # 2 for accidentals, 3 for clef
                gg[y0:y1,x0:x1] = sfnClef.getType()+2
    # will want the noteGroup to be "more dominent" than the sfnClef, so it overwrites if it's place of a sfn
    ys,xs = np.where(noteGroupMap>0)
    gg[ys,xs] = 1
    beamMapImg = cv2.merge([bb,gg,rr])
    return sfnClefList,sfnClefMap, beamMapImg

def findBarlines(image:np.ndarray, beamMapImg:np.ndarray, staffObjList:List[Staff], noteGroupStemMap:np.ndarray, thres:float)->np.ndarray:
    img200 =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    barlineImg = np.zeros((image.shape[0], image.shape[1]),dtype=np.uint8)
    # (1) where there's stuff, (0) where there's no
    _,img200bin = cv2.threshold(img200, 200, 1, cv2.THRESH_BINARY_INV) 
    ys,xs = np.where(beamMapImg[:,:,1]>0)
    img200bin[ys,xs] = 0
    ys,xs = np.where(noteGroupStemMap>0)
    img200bin[ys,xs] = 0
    if len(staffObjList)%NUM_TRACK!= 0:
        print(f"staff List is not divisible: {len(staffObjList)}%{NUM_TRACK}")
        for idx,staff in enumerate(staffObjList):
            ys = staff.ys
            y0 = ys[0]
            y1 = ys[-1]
            xs = np.where(np.sum(img200bin[y0:y1, :],0)>(y1-y0)*thres)[0]
            beamMapImg[y0:y1,xs,1] = 4
            beamMapImg[y0:y1,xs-1,1] = 4
            beamMapImg[y0:y1,xs+1,1] = 4
            barlineImg[y0:y1, xs] = 1
        contours, _ = cv2.findContours(barlineImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        barlineboxes:List[Tuple[int,int,int,int]] = []
        imgbgr = image.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            barlineboxes.append((x,y,x+w,y+h))
            imgbgr = cv2.rectangle(imgbgr, (x,y), (x+w, y+h), (0,0,255),1,cv2.LINE_AA)
    else:
        for linNo in range(len(staffObjList)//NUM_TRACK):
            y0 = staffObjList[NUM_TRACK*linNo].ys[0]
            y1 = staffObjList[NUM_TRACK*(linNo+1)-1].ys[-1]
            xs = np.where(np.sum(img200bin[y0:y1, :],0)>(y1-y0)*thres)[0]
            if xs[0] == 0:
                xs = xs[1:]
            while xs[-1] >= img200bin.shape[1]-1:
                xs = xs[:-1]
            beamMapImg[y0:y1,xs,1] = 4
            beamMapImg[y0:y1,xs-1,1] = 4
            beamMapImg[y0:y1,xs+1,1] = 4
            barlineImg[y0:y1, xs] = 1
        contours, _ = cv2.findContours(barlineImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        barlineboxes:List[Tuple[int,int,int,int]] = []
        imgbgr = image.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            barlineboxes.append((x,y,x+w,y+h))
            imgbgr = cv2.rectangle(imgbgr, (x,y), (x+w, y+h), (0,0,255),1,cv2.LINE_AA)
    imwrite("barLineImg.jpg",imgbgr)
    return beamMapImg, barlineboxes, imgbgr

def filterBarlineBoxes(image:np.ndarray, barlineboxes:List[Tuple[int,int,int,int]]):
    imgbgr = image.copy()
    img200 =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,img200bin = cv2.threshold(img200, 200, 1, cv2.THRESH_BINARY_INV) 
    for b in barlineboxes:
        thres = (b[3]-b[1])/4
        if np.sum(img200bin[b[1]:b[3],b[0]-2:b[0]])<thres*2 and np.sum(img200bin[b[1]:b[3],b[2]+1:b[2]+3])<thres*2:
            imgbgr = cv2.rectangle(imgbgr, (b[0],b[1]),(b[2],b[3]), (0,0,255),1,cv2.LINE_AA)
    return imgbgr

keyCharMapRest = {'z':'r4',
                  'x':'r8',
                  'c':'r16',
                  'v':'r32',
                  's':'X'}
def init_rest_folder(rootFolder = 'training'):
    rootFolder = 'training'
    foldername = ['rest_img','rest_stem','rest_remain']
    for fd in foldername:
        folderPath = os.path.join(rootFolder,fd)
        if not os.path.exists(folderPath):
            os.mkdir(folderPath)
            for one_dir in list(keyCharMapRest.values()):
                if not os.path.exists(os.path.join(folderPath,one_dir)):
                    os.mkdir(os.path.join(folderPath,one_dir))

def labelRestData(labelingImages:List[np.ndarray], bboxes:List[Tuple[int,int,int,int]], imgname:str):
    rootFolder = 'training'
    foldername = ['rest_img','rest_stem','rest_remain']
    for idx,b in enumerate(bboxes):
        cropImages = [img[b[1]:b[3],b[0]:b[2]] for img in labelingImages]
        cv2.imshow('t', np.hstack(cropImages))
        k=cv2.waitKey()
        cv2.destroyAllWindows()
        if chr(k) == 'a':
            print(f"stopping at img idx: {idx}")
            break
        else:
            filname = f'{imgname}_{idx}.jpg'
            classFolder = 'X'
            key_chr = chr(k)
            try:
                classFolder = keyCharMapRest[key_chr]
            except:
                print(f'invalid keychr: {key_chr}')
            for i in range(len(foldername)):
                cv2.imwrite(f'{rootFolder}/{foldername[i]}/{classFolder}/{filname}', cropImages[i])        

def findRests(dataDict:dict, 
              beamMapImg: np.ndarray,barheight:float, 
              model: Rest_Classifier,):
    restList:List[Rest|None] = [None]
    restMap = np.zeros((beamMapImg.shape[0], beamMapImg.shape[1]), dtype=np.uint16)
    symbol = dataDict['symbols'].astype(np.uint8) # 1 where there's symbols
    stem_rests = dataDict['stems_rests'].astype(np.uint8)
    bb,gg,rr = cv2.split(beamMapImg)
    mask = np.ones_like(symbol, dtype= np.uint8)
    stemRestClean = cv2.cvtColor(stem_rests*255, cv2.COLOR_GRAY2BGR)
    ys,xs = np.where(bb==255)
    mask[ys,xs] = 0
    stem_rests[ys,xs] = 0
    # mask = cv2.erode(mask.astype(np.uint8), np.ones((barheight//2, barheight//4),dtype=np.uint8)).squeeze()
    ys,xs = np.where(gg>0)
    mask[ys,xs] = 0
    stem_rests[ys,xs] = 0
    stem_rests = cv2.dilate(stem_rests.astype(np.uint8), np.ones((barheight//3, barheight//3), dtype= np.uint8))

    # remaining = cv2.bitwise_or(cv2.bitwise_and(mask,symbol), stem_rests)*255
    remaining = cv2.bitwise_and(mask,symbol)*255
    remaining = cv2.cvtColor(remaining,cv2.COLOR_GRAY2BGR)
    symbol = cv2.cvtColor(symbol*255,cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(stem_rests.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    stem_rests = cv2.cvtColor((stem_rests*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    # save the image version (later can give img128 version), stem_rests itself, remaining
    bboxes = []
    longBoxes = []
    # imageClean = image.copy()
    remainingClean = remaining.copy()
    class_colors = [(255,0,0),(0,0,255),(255,255,0),(0,120,255),(40,255,40), (245, 220, 255),(230,130,175),(165,170, 70)]
    restClassNames = ['X', 'r16', 'r32', 'r4', 'r8'] 
    restToRhythm = [0,2,3,0,1]
    imgrgb = dataDict['image'].copy()
    for j in range(0,len(restClassNames)):
        imgrgb = cv2.putText(imgrgb,restClassNames[j], (30, 30+j*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_colors[j], 2, cv2.LINE_AA)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w>barheight*0.6 and h>barheight):
            box = (max(x-barheight//2, 0), max(y-barheight*2, 0), min(x+w+barheight//2, symbol.shape[1]-1), min(y+h+barheight*2, symbol.shape[0]-1))
            # shrink the box so it centers around the rest
            yTopBlank = max(np.max(np.where(np.sum(remainingClean[box[1]:(box[1]+box[3])//2, box[0]:box[2],1],1)<255)[0].tolist()+[0])-barheight//4, 0)
            # the index of the ending (the new height)
            yBottomBlank = min(np.min(np.where(np.sum(remainingClean[(box[1]+box[3])//2:box[3], box[0]:box[2],1],1)<255)[0].tolist()+[box[3]-box[1]])+(box[3]-box[1])//2+barheight//4, box[3]-box[1])
            # xLeftBlank = max(np.max(np.where(np.sum(remainingClean[box[1]:box[3], box[0]:(box[0]+box[2])//2,1],0)<255)[0].tolist()+[0]), 0)
            # yRightBlank = min(np.min(np.where(np.sum(remainingClean[box[1]:box[3], (box[0]+box[2])//2:box[2],1],0)<255)[0].tolist()+[box[2]-box[0]])+(box[2]-box[0])//2, box[2]-box[0])
            # redo the training? No
            predict_idx = model.predict(remainingClean[box[1]:box[3], box[0]:box[2],:])
            # predict_idx = model.predict(remainingClean[box[1]+yTopBlank:box[1]+yBottomBlank, box[0]+xLeftBlank:box[0]+yRightBlank,:])
            box = (x, box[1]+yTopBlank, x+w, box[1]+yBottomBlank)
            if predict_idx>0:
                bboxes.append(box)
                restMap[box[1]:box[3],box[0]:box[2]] = len(restList)
                restList.append(Rest(box,restToRhythm[predict_idx]))
                gg = cv2.rectangle(gg, (box[0], box[1]), (box[2],box[3]),5, -1)
                imgrgb = cv2.rectangle(imgrgb,(box[0], box[1]), (box[2],box[3]), class_colors[predict_idx], 2, cv2.LINE_AA)
                # remaining = cv2.rectangle(remaining, (box[0], box[1]), (box[2],box[3]), (0,255,0), 2, cv2.LINE_AA)
                # stem_rests = cv2.rectangle(stem_rests, (box[0], box[1]), (box[2],box[3]), (0,255,0), 2, cv2.LINE_AA)
                # symbol = cv2.rectangle(symbol, (box[0], box[1]), (box[2],box[3]), (0,255,0), 2, cv2.LINE_AA)
        elif (w>barheight*0.8 and h<barheight*1.2 and h>barheight*0.3):
            box = (x,y,x+w,y+h)
            longBoxes.append(box)
            gg = cv2.rectangle(gg, (box[0], box[1]), (box[2],box[3]),5, -1)
            imgrgb = cv2.rectangle(imgrgb,(box[0], box[1]), (box[2],box[3]), (255,0,255), 2, cv2.LINE_AA)
            restMap[box[1]:box[3],box[0]:box[2]] = len(restList)
            restList.append(Rest(box,-1))
            # remaining = cv2.rectangle(remaining, (box[0], box[1]), (box[2],box[3]), (0,0,255), 2, cv2.LINE_AA)
            # stem_rests = cv2.rectangle(stem_rests, (box[0], box[1]), (box[2],box[3]), (0,0,255), 2, cv2.LINE_AA)
            # symbol = cv2.rectangle(symbol, (box[0], box[1]), (box[2],box[3]), (0,0,255), 2, cv2.LINE_AA)
    # init_rest_folder()            
    # labelRestData([imageClean, stemRestClean, remainingClean], bboxes, imgname)

    imwrite('remaining.jpg',remaining)
    imwrite('remainingStemRests.jpg',stem_rests)
    imwrite('RestClassification.jpg',imgrgb)
    beamMapImg = cv2.merge([bb,gg,rr])
    # return {'RestBarline1': remaining, 'RestBarline2': symbol}
    return restMap, restList, beamMapImg, {'RestClassification': imgrgb}

def assignPitch(image:np.ndarray, noteGroupList:List[NoteGroup|None], beamMapImg:np.ndarray, barheight:int):
    # assign the "pitch" relative to the center of the staff (12.5),
    # ex: in violin, B will be 0, C:1,D:2, A: -1 etc
    # if it doesn't have a staff line id yet, group it using the beamMapImg[:,:,4]
    # image is for outputting debug images
    pitchImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, imgBin = cv2.threshold(pitchImg, 200, 1, cv2.THRESH_BINARY_INV)
    _,pitchImg = cv2.threshold(pitchImg, 200, 255, cv2.THRESH_BINARY)
    pitchLineImg = pitchImg.copy()
    pitchLineImg = cv2.cvtColor(pitchLineImg, cv2.COLOR_GRAY2BGR)
    pitchImg = pitchImg//4+191
    pitchImg = cv2.cvtColor(pitchImg, cv2.COLOR_GRAY2BGR)
    pitch_colors = [(255,0,0),(0,0,255),(255,255,0),(0,120,255),(40,255,40), (255,0,255),(230,130,175),(165,170, 70)]
    pitch_names = ['B','C','D','E','F','G','A']
    for j in range(0,len(pitch_names)):
        pitchImg = cv2.putText(pitchImg,pitch_names[j], (30, 30+j*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pitch_colors[j], 2, cv2.LINE_AA)
    for ng in noteGroupList:
        if ng is None:
            continue
        for stemObj in ng.noteStemList:
            x0,y0,x1,y1 = stemObj.smallNoteBox
            staffNum = np.max(beamMapImg[y0:y1,x0:x1,2])
            if staffNum == 0:
                staffNum = round(np.sum(beamMapImg[y0:y1,4,2]/(y1-y0)))
            height = np.sum((beamMapImg[y0:y1,staffNum%4,2]-12.5)*2)/(y1-y0)
            stemObj.pitchFloat = height

            x0,y0,x1,y1 = stemObj.noteBox
            height2 = np.sum((beamMapImg[y0:y1,staffNum%4,2]-12.5)*2)/(y1-y0)
            if height2-int(height2) == 0:
                stemObj.pitchWideFloat = height
            else:
                stemObj.pitchWideFloat = height2

    pitchAndLineImg = pitchImg.copy()
    modifiedPitchImg = pitchImg.copy()

    for ngIdx, ng in enumerate(noteGroupList):
        if ng is None:
            continue
        for idx,stemObj in enumerate(ng.noteStemList):
            x0,y0,x1,y1 = stemObj.smallNoteBox
            _,y0B, _, y1B = stemObj.noteBox
            pitch = stemObj.pitchFloat
            xx0 = max(x0-barheight//2, 0)
            xx1 = min(x1+barheight//2, beamMapImg.shape[1]-1)
            cc = np.sum(imgBin[max(y0B,y0-2):min(y1B,y1+3), xx0:xx1],1).tolist()
            ccNozero = [c for c in cc if c>0]
            maxIdx = [i for i in range(len(ccNozero)) if ccNozero[i]==max(ccNozero)]
            # center that was tilted and thick
            if (maxIdx[0]>len(ccNozero)/2 and maxIdx[0]<len(ccNozero)*0.)or (maxIdx[-1]<len(ccNozero)/2 and maxIdx[-1]>len(ccNozero)*0.8) and maxIdx[-1]-maxIdx[0]>=3: # TODO!!!
                stemObj.hasLineMiddle = True
                pitchLineImg=cv2.rectangle(pitchLineImg,(xx0, y0+maxIdx[0]),(xx1, y0+maxIdx[-1]),(255,0,255), 2, cv2.LINE_AA)
            # center
            if len(set.intersection(set([len(ccNozero)//2,len(ccNozero)//2-1, len(ccNozero)//2+1]), maxIdx))>0:
                stemObj.hasLineMiddle = True
                pitchLineImg=cv2.rectangle(pitchLineImg,(xx0, y0+maxIdx[0]),(xx1, y0+maxIdx[-1]),(255,0,255), 2, cv2.LINE_AA)
            # top or bottom
            elif y1-y0>barheight//2 and (maxIdx[0]<=3 or maxIdx[-1]>=len(ccNozero)-4):
                stemObj.hasLineMiddle = False
                pitchLineImg=cv2.rectangle(pitchLineImg,(xx0, y0),(xx1, y1),(0,255,0), 1, cv2.LINE_AA)
            # if it has line in the middle
            elif ccNozero[maxIdx[0]]>(x1-x0):
                stemObj.hasLineMiddle = True
                pitchLineImg=cv2.rectangle(pitchLineImg,(xx0, y0+maxIdx[0]),(xx1, y0+maxIdx[-1]),(255,0,255), 2, cv2.LINE_AA)
            else:
                stemObj.hasLineMiddle = True
                pitchLineImg=cv2.rectangle(pitchLineImg,(xx0, y0+maxIdx[0]),(xx1, y0+maxIdx[-1]),(255,255,0), 2, cv2.LINE_AA)

            p1 = math.ceil(pitch)
            p2 = math.floor(pitch)
            pitchLineImg = cv2.putText(pitchLineImg, str(ngIdx), (x0,y0-barheight*2), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,0), 1, cv2.LINE_AA)
            pitchAndLineImg = cv2.putText(pitchAndLineImg, str(ngIdx), (x0,y0-barheight*2), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,0), 1, cv2.LINE_AA)
            pitchAndLineImg = cv2.putText(pitchAndLineImg, str(pitch_names[p1%7]), (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pitch_colors[p1%7], 2, cv2.LINE_AA)
            pitchAndLineImg = cv2.putText(pitchAndLineImg, str(pitch_names[p2%7]), (x0,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pitch_colors[p2%7], 2, cv2.LINE_AA)
            pitchAndLineImg = cv2.rectangle(pitchAndLineImg, (x0,y0),(x1,y1), (30,30,30), 1,cv2.LINE_AA)
            pitchImg = cv2.putText(pitchImg, str(pitch_names[round(pitch)%7]), (x0,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pitch_colors[round(pitch)%7], 2, cv2.LINE_AA)
            pitchImg = cv2.rectangle(pitchImg, (x0,y0),(x1,y1), (30,30,30), 1,cv2.LINE_AA)
            
    xend = pitchLineImg.shape[1]-1
    for i in range(pitchLineImg.shape[0]-1):
        currStaffNum = beamMapImg[i,4,2]
        if beamMapImg[i,currStaffNum%4, 2] != beamMapImg[i+1,currStaffNum%4, 2]:
            pitchLineImg = cv2.line(pitchLineImg,(0,i),(xend,i),pitch_colors[currStaffNum%4],thickness=1, lineType=cv2.LINE_AA)
            pitchAndLineImg = cv2.line(pitchAndLineImg,(0,i),(xend,i),pitch_colors[currStaffNum%4],thickness=1, lineType=cv2.LINE_AA)
    
    for idx, ng in enumerate(noteGroupList):
        if ng is None:
            continue
        for stem in ng.noteStemList:
            pitchInt = stem.getBestPitchInt()
            x0,y0,x1,y1 = stem.noteBox
            # modifiedPitchImg = cv2.putText(modifiedPitchImg, str(idx), (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pitch_colors[pitchInt%7], 2, cv2.LINE_AA)
            modifiedPitchImg = cv2.putText(modifiedPitchImg, str(pitch_names[pitchInt%7]), (x0,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pitch_colors[pitchInt%7], 2, cv2.LINE_AA)
            modifiedPitchImg = cv2.rectangle(modifiedPitchImg, (x0,y0),(x1,y1), (30,30,30), 1,cv2.LINE_AA)
            stem.pitchSoprano = pitchInt
        
    
    imwrite('pitchline.jpg', pitchLineImg)
    imwrite('pitchAndLine.jpg', pitchAndLineImg)
    imwrite('pitch.jpg', pitchImg)
    imwrite('pitchModified.jpg', modifiedPitchImg)
    return noteGroupList, {'pitch':pitchImg, 
                           'pitchAndLine':pitchAndLineImg, 
                           'pitchline': pitchLineImg,
                           'pitchModified':modifiedPitchImg}

def enhanceStaff(image:np.ndarray, staffObjList:List[Staff],barheight:int):
    _, img200 = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    for staff in staffObjList:
        if staff.IsStaffAligned(barheight):
            for y in staff.ys:
                img200 = cv2.line(img200, (staff.left, y), (staff.right, y), (0,0,0), 1, cv2.LINE_AA)
    imwrite('enhanceStaff.jpg',img200)
    return img200

def assignDots(noteGroupVerticallyMerged:List[NoteGroup],beamMapImg:np.ndarray, image:np.ndarray,bar_height:int,StaffObjList:List[Staff]):
    global DEBUGIMG
    imgbgr = image.copy() #bgr
    _,gg,_ = cv2.split(beamMapImg)
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = bar_height*bar_height//16
    params.maxArea = bar_height*bar_height//2

    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = True
    params.minInertiaRatio = 0.5
    
    detector = cv2.SimpleBlobDetector_create(params)
    _, thresh_image = cv2.threshold(image[:,:,1], 150, 255, cv2.THRESH_BINARY)
    thresh_origin = thresh_image.copy()
    for sf in staffObjList:
        for yOne in sf.ys:
            thresh_image[yOne-math.ceil(bar_height/8):yOne+bar_height//8+1,:] = 255

    for ng in noteGroupVerticallyMerged:
        if ng is None:
            continue
        for sm in ng.noteStemList:
            oldbox = sm.noteBox
            newbox = (oldbox[2],max(0,oldbox[1]-2*bar_height),
                      min(oldbox[2]+bar_height, image.shape[1]-1), min(oldbox[1]+2*bar_height, image.shape[0]-1))
            realWidth = np.max(np.append(np.where(np.max(gg[newbox[1]:newbox[3],newbox[0]:newbox[2]],0)==0)[0],0))
            # since the barline usually isn't thick enough
            realWidth = min(realWidth, np.min(np.append(np.where(np.max(gg[newbox[1]:newbox[3],newbox[0]:newbox[2]],0)==4)[0],realWidth)))
            if realWidth>=bar_height-1:
                realWidth = np.max(np.append(np.where(np.max(gg[newbox[1]:newbox[3],newbox[0]:min(newbox[2]+bar_height,image.shape[1]-1)],0)==0)[0],0))
                realWidth = min(realWidth, np.min(np.append(np.where(np.max(gg[newbox[1]:newbox[3],newbox[0]:min(newbox[2]+bar_height,image.shape[1]-1)],0)==4)[0],realWidth)))
            if realWidth>bar_height*0.5:
                dotBox = (oldbox[2], max(0,oldbox[1]-bar_height),
                        min(oldbox[2]+realWidth,image.shape[1]-1),min(oldbox[1]+int(bar_height*1.5), image.shape[0]-1))
                keypoints = detector.detect(thresh_image[dotBox[1]:dotBox[3], dotBox[0]:dotBox[2]])
                if len(keypoints)==0:
                    imgbgr = cv2.rectangle(imgbgr,(dotBox[0],dotBox[1]),(dotBox[2],dotBox[3]),(0,255,255),1,cv2.LINE_AA)
                elif (np.where(np.min(thresh_origin[dotBox[1]:dotBox[3], dotBox[0]+bar_height//2:dotBox[2]],1)==255)[0]).shape[0]>0 and np.min(np.where(np.min(thresh_origin[dotBox[1]:dotBox[3], dotBox[0]+bar_height//2:dotBox[2]],1)==255))>bar_height:
                    DEBUGIMG = cv2.rectangle(DEBUGIMG,(dotBox[0],dotBox[1]),(dotBox[2],dotBox[3]),(160,100,240),2,cv2.LINE_AA)
                    imgbgr = cv2.rectangle(imgbgr,(dotBox[0],dotBox[1]),(dotBox[2],dotBox[3]),(160,100,240),2,cv2.LINE_AA)
                else:
                    sm.hasdot = True
                    imgbgr = cv2.rectangle(imgbgr,(dotBox[0],dotBox[1]),(dotBox[2],dotBox[3]),(255,255,0),2,cv2.LINE_AA)
                    DEBUGIMG = cv2.rectangle(DEBUGIMG,(dotBox[0],dotBox[1]),(dotBox[2],dotBox[3]),(255,255,0),2,cv2.LINE_AA)

    imwrite('dotBox.jpg',imgbgr)
    return noteGroupVerticallyMerged,{'dotBox':imgbgr}

def assignRestDots(restList:List[Rest], beamMapImg:np.ndarray, noteGroupStemMap:np.ndarray, image:np.ndarray, bar_height:int, staffObjList:List[Staff]):
    imgbgr = image.copy() #bgr
    _,gg,_ = cv2.split(beamMapImg)
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = bar_height*bar_height//16
    params.maxArea = bar_height*bar_height//2

    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = True
    params.minInertiaRatio = 0.5
    
    detector = cv2.SimpleBlobDetector_create(params)
    _, thresh_image = cv2.threshold(image[:,:,1], 150, 255, cv2.THRESH_BINARY)
    thresh_origin = thresh_image.copy()
    for sf in staffObjList:
        for yOne in sf.ys:
            thresh_image[yOne-math.ceil(bar_height/8):yOne+bar_height//8+1,:] = 255
    for rs in restList:
        if rs is None:
            continue
        elif rs.rhythm<0:
            continue
        newbox = (rs.boundingBox[2],rs.boundingBox[1],
                    min(rs.boundingBox[2]+bar_height,image.shape[1]),min(rs.boundingBox[1]+2*bar_height,image.shape[0]))
        # 先用一個barheight寬度，高度從最高點往下兩個barheight，往上一個，用stem跟rest來縮減
        realWidth = np.max(np.append(np.where(np.max(noteGroupStemMap[newbox[1]:newbox[3],newbox[0]:newbox[2]],0)==0)[0],0))
        realWidth = min(realWidth, np.max(np.append(np.where(np.max(beamMapImg[newbox[1]:newbox[3],newbox[0]:newbox[2]],0)!=5)[0],0)))
        realWidth = min(realWidth, np.min(np.append(np.where(np.max(gg[newbox[1]:newbox[3],newbox[0]:newbox[2]],0)==4)[0],realWidth)))
        # 如果寬度還是滿格的話再往右延長到2*barheight用beamMapImg 來縮減
        if realWidth>=bar_height-1:
            realWidth = np.max(np.append(np.where(np.max(gg[newbox[1]:newbox[3],newbox[0]:min(newbox[2]+bar_height,image.shape[1]-1)],0)==0)[0],0))
            realWidth = min(realWidth, np.min(np.append(np.where(np.max(gg[newbox[1]:newbox[3],newbox[0]:min(newbox[2]+bar_height,image.shape[1]-1)],0)==4)[0],realWidth)))
        elif realWidth>bar_height*0.5:
            realWidth = realWidth+bar_height//3
        if realWidth>bar_height*0.5:
            dotBox = newbox
            keypoints = detector.detect(thresh_image[dotBox[1]:dotBox[3], dotBox[0]:dotBox[2]])
            if len(keypoints)==0:
                imgbgr = cv2.rectangle(imgbgr,(dotBox[0],dotBox[1]),(dotBox[2],dotBox[3]),(0,255,200),1,cv2.LINE_AA)
            elif len(np.where(np.min(thresh_origin[dotBox[1]:dotBox[3], dotBox[0]+bar_height//2:dotBox[2]],1)==255)[0]) == 0:
                imgbgr = cv2.rectangle(imgbgr,(dotBox[0],dotBox[1]),(dotBox[2],dotBox[3]),(255,255,0),2,cv2.LINE_AA)
            elif np.min(np.where(np.min(thresh_origin[dotBox[1]:dotBox[3], dotBox[0]+bar_height//2:dotBox[2]],1)==255))>bar_height:
                imgbgr = cv2.rectangle(imgbgr,(dotBox[0],dotBox[1]),(dotBox[2],dotBox[3]),(160,100,240),2,cv2.LINE_AA)
            else:
                imgbgr = cv2.rectangle(imgbgr,(dotBox[0],dotBox[1]),(dotBox[2],dotBox[3]),(255,255,0),2,cv2.LINE_AA)
                rs.hasdot = True
    imwrite('dotRestBox.jpg',imgbgr)
    return restList,{'dotRestBox':imgbgr}
def getNoteChunks(image: np.ndarray, noteGroupMap:np.ndarray, beamMapImg:np.ndarray, noteGroupVerticallyMerged:List[NoteGroup]):
    imgrgb = image.copy()
    beamBinary = np.zeros((beamMapImg.shape[0],beamMapImg.shape[1]))
    xs,ys = np.where(beamMapImg[:,:,0]==255)
    beamBinary[xs,ys] = 1
    contours, _ = cv2.findContours(beamBinary.astype(np.uint8),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    beamBoxes:List[Tuple[int,int,int,int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        beamBoxes.append((x,y,x+w,y+h))
        imgrgb = cv2.rectangle(imgrgb,(x,y), (x+w, y+h), (0,0,255), 2, cv2.LINE_AA)
    imwrite('beamContour.jpg',imgrgb)

    noteChunkList:List[NoteChunk] = [None]
    noteChunkMap = np.zeros_like(noteGroupMap)
    for box in beamBoxes:
        currentNoteChunkIdxs = set(np.unique(noteChunkMap[box[1]:box[3], box[0]:box[2]]))
        allNoteGroupIdxs = set(np.unique(noteGroupMap[box[1]:box[3], box[0]:box[2]]))
        allNoteGroupIdxs.add(0)
        allNoteGroupIdxs.remove(0)
        currentNoteChunkIdxs.add(0)
        currentNoteChunkIdxs.remove(0)
        if len(allNoteGroupIdxs) == 0:
            continue
        if len(currentNoteChunkIdxs)==0:
            n = NoteChunk(allNoteGroupIdxs)
            for a in allNoteGroupIdxs:
                ys,xs = np.where(noteGroupMap==a)
                noteChunkMap[ys,xs] = len(noteChunkList)
            noteChunkList.append(n)
        elif len(currentNoteChunkIdxs) == 1:
            noteChunkList[min(currentNoteChunkIdxs)].mergeNoteChunk(allNoteGroupIdxs)
            for a in allNoteGroupIdxs:
                ys,xs = np.where(noteGroupMap==a)
                noteChunkMap[ys,xs] = min(currentNoteChunkIdxs)
        else:
            minChunkIdx = min(currentNoteChunkIdxs)
            currentNoteChunkIdxs.remove(minChunkIdx)
            for chunkIdx in list(currentNoteChunkIdxs):
                ys,xs = np.where(noteChunkMap==chunkIdx)
                noteChunkMap[ys,xs] = minChunkIdx
                noteChunkList[minChunkIdx].mergeNoteChunk(noteChunkList[chunkIdx].noteGroupIdxs)
                noteChunkList[chunkIdx] = None
    imggg = image.copy()
    for ii, nc in enumerate(noteChunkList):
        if nc is None:
            continue
        for id in nc.noteGroupIdxs:
            noteGroupVerticallyMerged[id].noteChunkId = ii
        bx = nc.getBoundingBox(noteGroupList)
        imggg = cv2.rectangle(imggg, (bx[0],bx[1]),(bx[2], bx[3]),(0,0,255), 2, cv2.LINE_AA)
    imwrite("horizontalGrouping.jpg", imggg)
    return noteChunkList, imggg
    
def getMaskedImageTimeSignature(image:np.ndarray, beamMapImg: np.ndarray, staffList:List[Staff], barheight: int,img_name:str):
    bb,gg,_ = cv2.split(beamMapImg)
    imgMasked = image.copy()
    topBound = 0
    for sf in staffList:
        lowerBound = sf.ys[0]
        imgMasked = cv2.rectangle(imgMasked, (0, topBound),(imgMasked.shape[1], lowerBound), (255,255,255), -1)
        topBound = sf.ys[-1]
        # imgCrop = image[sf.ys[0]+1:sf.ys[-1],:,:]
        # cv2.imwrite(f'TimeSignature/tsLong/{img_name}{topBound}{lowerBound}.jpg', imgCrop)
    lowerBound = imgMasked.shape[0]
    imgMasked = cv2.rectangle(imgMasked, (0, topBound),(imgMasked.shape[1], lowerBound), (255,255,255), -1)
    imwrite('imgMasked.jpg', imgMasked)
    return imgMasked

def getCTimeSignature(image:np.ndarray, imgMasked:np.ndarray, staffList:List[Staff]):
    tsCImage = image.copy()
    # create a matrix of zeros for later (add 1 for each matching boxes and then count + threshold)
    tsCount = np.zeros((image.shape[0], image.shape[1]))
    match_CThick = cv2.imread("training/CpatternThick.jpg",cv2.IMREAD_GRAYSCALE)
    match_COri = cv2.imread("training/CpatternLong.jpg",cv2.IMREAD_GRAYSCALE)
    for staff in staffList:
        sf = staff.ys
        seg_img = image[sf[0]:sf[-1],:,1]
        resize_ratio = seg_img.shape[0]/match_CThick.shape[0]
        match_c = cv2.resize(match_CThick, None, fx=resize_ratio, fy=resize_ratio, interpolation = cv2.INTER_AREA)
        newHeight = match_c.shape[0]
        match_c = match_c[newHeight//12:newHeight-newHeight//12]
        res1 = cv2.matchTemplate(seg_img, match_c, cv2.TM_SQDIFF_NORMED)

        resize_ratio2 = seg_img.shape[0]/match_COri.shape[0]
        match_c2 = cv2.resize(match_COri, None, fx=resize_ratio2, fy=resize_ratio2, interpolation = cv2.INTER_AREA)
        newHeight2 = match_c2.shape[0]
        match_c2 = match_c2[newHeight2//12:newHeight2-newHeight2//12]
        res2 = cv2.matchTemplate(seg_img, match_c2, cv2.TM_SQDIFF_NORMED)
        resLst = [res1, res2]
        refLst = [match_c, match_c2]

        for idx, res in enumerate(resLst):
            w,h = refLst[idx].shape[::-1]
            threshold = 0.15
            loc = np.where(res <= threshold)
            if len(loc[0])>0:
                for pt in zip(*loc[::-1]):
                    tsCount[pt[1]+sf[0]:pt[1]+sf[0]+h, pt[0]:pt[0]+w]+=1
                    # tsCImage = cv2.rectangle(tsCImage, (pt[0],pt[1]+sf[0]), (pt[0] + w, pt[1] + h + sf[0]), (0,0,255), 1)
    
    _,cPositionFiltered = cv2.threshold(tsCount, 5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(cPositionFiltered.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        tsCImage = cv2.rectangle(tsCImage, (x,y), (x + w,y+h), (0,0,255), 2)
    imwrite('tsCImage.jpg',tsCImage)
    return tsCImage
    
def getTimeSignature4(image:np.ndarray, imgMasked:np.ndarray, staffList:List[Staff]):
    tsCImage = image.copy()
    # create a matrix of zeros for later (add 1 for each matching boxes and then count + threshold)
    tsCount = np.zeros((image.shape[0], image.shape[1]))
    match_8Ori = cv2.imread("training/8Pattern.jpg",cv2.IMREAD_GRAYSCALE)
    match_4Ori = cv2.imread("training/4Pattern4x3.jpg",cv2.IMREAD_GRAYSCALE)
    thres = [0.15, 0.13]
    for staff in staffList:
        sf = staff.ys
        seg_img = image[sf[2]:sf[-1]+1,:,1]
        for currId, match_ori in enumerate([match_4Ori, match_8Ori]):
            resize_ratio = seg_img.shape[0]/match_ori.shape[0]
            match_c = cv2.resize(match_ori, None, fx=resize_ratio, fy=resize_ratio, interpolation = cv2.INTER_AREA)
            newHeight = match_c.shape[0]
            match_c = match_c[newHeight//12:newHeight-newHeight//12]
            res1 = cv2.matchTemplate(seg_img, match_c, cv2.TM_SQDIFF_NORMED)

            resLst = [res1]
            refLst = [match_c]

            for idx, res in enumerate(resLst):
                w,h = refLst[idx].shape[::-1]
                threshold = thres[currId]
                loc = np.where(res <= threshold)
                if len(loc[0])>0:
                    for pt in zip(*loc[::-1]):
                        tsCount[pt[1]+sf[0]:pt[1]+sf[2]+h, pt[0]:pt[0]+w]+=1
                        # tsCImage = cv2.rectangle(tsCImage, (pt[0],pt[1]+sf[0]), (pt[0] + w, pt[1] + h + sf[0]), (0,0,255), 1)
    
    _,cPositionFiltered = cv2.threshold(tsCount, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(cPositionFiltered.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        tsCImage = cv2.rectangle(tsCImage, (x,y), (x + w,y+h), (0,0,255), 2)
    imwrite('ts4Image.jpg',tsCImage)
    return tsCImage

def assignSfnToNote(image:np.ndarray, noteGroupMap:np.ndarray, noteGroupVerticallyMerged:List[NoteGroup],sfnClefMap:np.ndarray,sfnClefList:List[Union[Accidentals, Clef, None]]):
    stemIdxMap = np.ones((noteGroupMap.shape[0], noteGroupMap.shape[1]), dtype=np.uint8)*(-1)
    accidentalsImg = image.copy()
    for ng in noteGroupVerticallyMerged:
        if ng is None:
            continue
        for idx, stem in enumerate(ng.noteStemList):
            x0,y0,x1,y1 = stem.noteBox
            stemIdxMap[y0:y1, x0:x1] = idx
    sfnGroupList = []
    for currSfnIdx, sfnc in enumerate(sfnClefList):
        if sfnc is None:
            continue
        if sfnc.getType() == 1: # is Clef
            continue
        sfnc:Accidentals
        x0,y_0,x1,y_1 = sfnc.getBbox() # y0: smaller (top), y1: bigger (bottom)
        if sfnc.shift >=0: # sharp or natural
            y0 = (y_0+y_1)//2-(y_1-y_0)//6
            y1 = (y_0+y_1)//2+(y_1-y_0)//6
        else:
            y0 = (y_0+y_1)//2
            y1 = y_1
        sfnc.shrinkYs = (y0,y1)
        stemIdxLst = np.unique(stemIdxMap[y0:y1,x1:x1+(x1-x0)]).tolist()
        if -1 in stemIdxLst: 
            stemIdxLst.remove(-1)
        ngIdxLst = np.unique(noteGroupMap[y0:y1,x1:x1+(x1-x0)]).tolist()
        if 0 in ngIdxLst:
            ngIdxLst.remove(0)
        if len(stemIdxLst)==0 or len(ngIdxLst)==0:
            # is Key signature or has overlapping sfns
            sfnList = np.unique(sfnClefMap[y_0:y_1,x1:x1+(x1-x0)//2]).tolist()
            if sfnList == [0]:
                sfnc.isKeySignature = True
            else:
                if 0 in sfnList:
                    sfnList.remove(0)
                sfnNext = sfnList[0] # sfnNext is the index of the next sfn
                if type(sfnClefList[sfnNext]) is not Accidentals:
                    continue
                hasAssigned = False
                for sIdx in range(len(sfnGroupList)):
                    if sfnNext in sfnGroupList[sIdx]:
                        sfnGroupList[sIdx] = [currSfnIdx]+sfnGroupList[sIdx]
                        hasAssigned = True
                        continue
                    elif currSfnIdx in sfnGroupList[sIdx]:
                        sfnGroupList[sIdx] = sfnGroupList[sIdx]+[sfnNext]
                        hasAssigned = True
                        continue
                if not hasAssigned:
                    sfnGroupList.append([currSfnIdx, sfnNext])
            continue
        stemIdx = stemIdxLst[0]
        ngIdx = ngIdxLst[0]
        if len(ngIdxLst)>1:                    
            cntLst = [np.sum(noteGroupMap[y0:y1,x1:x1+(x1-x0)]==i) for i in ngIdxLst]
            ngIdx = ngIdxLst[cntLst.index(max(cntLst))]
        if len(stemIdxLst)>1:   
            ngx0, _, ngx1, _ = noteGroupVerticallyMerged[ngIdx].boundingBox
            cntLst = [np.sum(np.sum(stemIdxMap[y0:y1,ngx0:ngx1]==i,0)>0) for i in stemIdxLst]
            stemIdx = stemIdxLst[cntLst.index(max(cntLst))]
        noteGroupVerticallyMerged[ngIdx].noteStemList[stemIdx].accidentals = sfnc.getValue()
        sfnc.ngIndex = ngIdx
    for grps in sfnGroupList:
        currNgIdx = sfnClefList[grps[-1]].ngIndex
        if currNgIdx is None:
            sfnClefList[grps[-1]].endKeySignature = True 
            for grr in grps:
                sfnClefList[grr].isKeySignature = True
            continue
        currNg:NoteGroup = noteGroupVerticallyMerged[currNgIdx]
        if len(currNg.noteStemList)==1:
            continue
        ngx0, _, ngx1, _ = currNg.boundingBox
        for grr in grps[:-1]:
            currSfn = sfnClefList[grr]
            y0,y1 = currSfn.shrinkYs
            stemIdxLst = np.unique(stemIdxMap[y0:y1,ngx0:ngx1]).tolist()
            if -1 in stemIdxLst: 
                stemIdxLst.remove(-1)
            if len(stemIdxLst)>=1:                    
                cntLst = [np.sum(np.sum(stemIdxMap[y0:y1,ngx0:ngx1]==i,0)>0) for i in stemIdxLst]
                stemIdx = stemIdxLst[cntLst.index(max(cntLst))]
                if currNg.noteStemList[stemIdx].accidentals is None:
                    noteGroupVerticallyMerged[currNgIdx].noteStemList[stemIdx].accidentals = currSfn.getValue()
                    noteGroupVerticallyMerged[currNgIdx].noteStemList[stemIdx].accidentalBox = currSfn.boundingBox
                else:
                    print('Error')


    accidentalsColors = [(255,255,0),(255,0,125),(255,0,255)] # flat, natural, sharp
    for ng in noteGroupVerticallyMerged:
        if ng is None:
            continue
        for idx, stem in enumerate(ng.noteStemList):
            if stem.accidentals is not None:
                x0,y0,x1,y1 = stem.noteBox
                currColor = accidentalsColors[stem.accidentals+1]
                accidentalsImg = cv2.rectangle(accidentalsImg, (x0,y0),(x1,y1),currColor,3, cv2.LINE_AA)
    imwrite("assignedAccidentals.jpg",accidentalsImg,strictlyYes=True)
    return stemIdxMap, noteGroupVerticallyMerged, {'assignedAccidentals':accidentalsImg}

def constructBar(noteGroupMap:np.ndarray, 
                   stemIdxMap: np.ndarray,
                   noteGroupVerticallyMerged: List[NoteGroup|None],
                   restMap: np.ndarray,
                   restList:List[Rest|None],
                   sfnClefMap:np.ndarray,
                   sfnClefList:List[Union[Accidentals,Clef, None]],
                   beamMapImg:np.ndarray,
                   staffList:List[Staff]):
    #TODO
    bb,gg,rr = cv2.split(beamMapImg)
    barList:List[List[Bar]] = [[] for _ in range(NUM_TRACK)]
    mapForMatching:List[np.ndarray|None] = [None,noteGroupMap,sfnClefMap, sfnClefMap, None, restMap]
    listForMatching:List[List] = [None, noteGroupVerticallyMerged,sfnClefList, sfnClefList, None, restList]
    notFinishedBar = None
    if len(staffList)%NUM_TRACK != 0:
        print(f"staff List is not divisible: {len(staffList)}%{NUM_TRACK}")
    numBarsEachLine = []
    allRanges = []
    for kk in range(len(staffList)//NUM_TRACK):
        allBars = []
        for j in range(NUM_TRACK):
            sf0 = staffList[kk*NUM_TRACK+j]
            barPlace = np.unique(np.where(gg[sf0.ys[0]:sf0.ys[-1], sf0.left:]==4)[1])
            allBars.append(barPlace)
        barsFlat = Counter(np.concatenate(allBars))
        isBar = np.sort(np.array([k for k, v in barsFlat.items() if v > 1]))
        ranges = []
        points = [0,isBar[0]] 
        i = 1
        while i < len(isBar):
            while isBar[i] == isBar[i-1]+1 and i<len(isBar)-1:
                i+=1
            if i == len(isBar)-1 and isBar[i] == isBar[i-1]+1:
                break
            points.append(isBar[i])
            i+=1
        for i in range(len(points)-1):
            if points[i+1]>points[i]+sf0.get_yOne():
                ranges.append((points[i]+sf0.left, points[i+1]+sf0.left))
        allRanges.append(ranges)
    for t in range(NUM_TRACK): # trackNo
        for j in range(len(staffList)//NUM_TRACK): # number of lines 
            printedLineNo = j*NUM_TRACK+t # the 
            currBarList:List[Bar] = []
            sf0 = staffList[printedLineNo]
            ranges = allRanges[j]
            for ridx, rng in enumerate(ranges):
                if notFinishedBar is not None:
                    bar = notFinishedBar
                    notFinishedBar = None
                else:
                    bar = Bar()
                currX = rng[0]
                while currX<rng[1]:
                    lineLst = np.unique(gg[sf0.ys[0]-sf0.get_yOne():sf0.ys[-1]+sf0.get_yOne(), currX]).tolist()
                    if 0 in lineLst:
                        lineLst.remove(0)
                    if len(lineLst)>1:
                        for i in [1,5,3,2]:
                            if i in lineLst:
                                lineLst = [i]
                                break
                    if len(lineLst)==0 or 4 in lineLst:
                        currX+=1
                    else:
                        typeId = lineLst[0]
                        currMap = mapForMatching[typeId]
                        currLst = listForMatching[typeId]
                        inMapId = np.unique(currMap[sf0.ys[0]-sf0.get_yOne():sf0.ys[-1]+sf0.get_yOne(), currX]).tolist()
                        if 0 in inMapId:
                            inMapId.remove(0)
                        if len(inMapId)==1: 
                            sanityCheck = True
                            if typeId == 3: # clef
                                currClef:Clef = sfnClefList[inMapId[0]]
                                _,yy0,_,yy1 = currClef.boundingBox
                                if yy1-yy0 < sf0.get_yOne_float()*3:
                                    sanityCheck = False
                                if currClef.type ==0:
                                    if yy0<sf0.ys[0]-sf0.get_yOne()/3:
                                        currClef.type = -2
                            if not sanityCheck:
                                # don't add the element if it's suspicious
                                currX = currLst[inMapId[0]].boundingBox[2]+1
                            elif typeId == 1:
                                x0,y0,x1,y1 = currLst[inMapId[0]].boundingBox
                                uniq = np.unique(rr[y0:y1, x0:x1]).tolist()
                                if 0 in uniq:
                                    uniq.remove(0)
                                if len(uniq) != 1:
                                    currX+=1
                                elif uniq[0] != printedLineNo+1:
                                    currX = currLst[inMapId[0]].boundingBox[2]+1
                                else:
                                    bar.addElement(currLst[inMapId[0]])
                                    if (currX == currLst[inMapId[0]].boundingBox[2]+1):
                                        print()
                                    currX = currLst[inMapId[0]].boundingBox[2]+1
                            else:
                                bar.addElement(currLst[inMapId[0]])
                                currX = currLst[inMapId[0]].boundingBox[2]+1
                        else:
                            currX+=1
                            print("has more than 1 id")
                currBarList.append(bar)
            
            nextIdx = np.max(np.unique(np.where(gg[sf0.ys[0]-sf0.get_yOne():sf0.ys[-1]+sf0.get_yOne(), rng[1]:]>0)[1]))
            if nextIdx>5:
                rng = (isBar[-1]+sf0.left+1, rng[1]+nextIdx)
                bar = Bar()
                currX = rng[0]
                while currX<rng[1]:
                    lineLst = np.unique(gg[sf0.ys[0]-sf0.get_yOne():sf0.ys[-1]+sf0.get_yOne(), currX]).tolist()
                    if 0 in lineLst:
                        lineLst.remove(0)
                    if len(lineLst)>1:
                        for i in [1,5,3,2]:
                            if i in lineLst:
                                lineLst = [i]
                                break
                    if len(lineLst)==0 or 4 in lineLst:
                        currX+=1
                    else:
                        typeId = lineLst[0]
                        currMap = mapForMatching[typeId]
                        currLst = listForMatching[typeId]
                        inMapId = np.unique(currMap[sf0.ys[0]-sf0.get_yOne():sf0.ys[-1]+sf0.get_yOne(), currX]).tolist()
                        if 0 in inMapId:
                            inMapId.remove(0)
                        if len(inMapId)==1:
                            bar.addElement(currLst[inMapId[0]])
                            currX = currLst[inMapId[0]].boundingBox[2]+1
                        else:
                            currX+=1
                            print("has more than 1 id")
                notFinishedBar = bar
            barList[t]+=currBarList
            if t == 0:
                numBarsEachLine.append(len(currBarList))
            # in order of vln1's 1st line, 2nd line ... | vln2's 1st line, 2nd line ...
    return barList,allRanges, numBarsEachLine

def exportXML(barList:List[List[Bar]], numTrack:int, image:np.ndarray|None = None, 
              beamMapImg:np.ndarray|None = None,
              barsBreakPoints: List[int] = [0],
              beamMapList:List[np.ndarray]|None = None,
              beamMapRefList:List[int]|None = None,
              lineNoList:List[int] | None = None):
    # Stem's label -> rhythm meaning
    # class_colors = [(255,0,0),(0,0,255),(255,255,0),(0,120,255),(40,255,40), (245, 220, 255),(230,130,175),(165,170, 70)]
    if image is None:
        debugXMLImg = None
    else:
        debugXMLImg = image.copy()
    if beamMapImg is not None:
        bb,gg,rr = cv2.split(beamMapImg)
    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = "Example MusicXML"
    score2 = stream.Score() # for those with shift
    score2.metadata = metadata.Metadata()
    score2.metadata.title = "Example MusicXML"

    part = [stream.Part() for n in range(numTrack)]
    part2 = [stream.Part() for n in range(numTrack)]
    keyName = ['C','D','E','F','G','A','B']
    keyCharFlat = ['C','D-','D','E-','E','F','G-','G','A-','A','B-','B']
    keyCharSharp = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    class_names = ['1/4','1/8','1/16','1/32', '1/64', '1/128', '1/2', '1/1']
    restClassNames = ['1/4','1/8','1/16','1/32'] # only for >=0
    currentClef = [None]*numTrack
    sharps = [3,0,4,1,5,2,6] # F C G D A E B
    flats = [6,2,5,1,4,0,3] # B E A D G C F
    ksSharp = [set() for _ in range(numTrack)]
    ksFlat = [set() for _ in range(numTrack)]
    currentKs = [0 for n in range(numTrack)]
    trksOriginal = [[] for _ in range (numTrack)] # for tracking all original notes we see
    trksShifted = [[] for _ in range (numTrack)] # for tracking all shifted notes
    def setKS(ks:float, trkNo):
        ks = int(ks)
        nonlocal ksSharp
        nonlocal ksFlat
        if ks>0:
            ksSharp[trkNo] = [keyName[i] for i in [sharps[q] for q in range(ks)]]
            ksFlat[trkNo] = set()
        else:
            ksFlat[trkNo] = [keyName[i] for i in [flats[q] for q in range(-ks)]]
            ksSharp[trkNo] = set()
    def setKSNeutral(Oriks:float, trkNo, trackShift):
        Oriks = int(Oriks)
        nonlocal ksSharp
        nonlocal ksFlat
        ks = Oriks+trackShift
        if ks>0:
            ksSharp[trkNo] = [keyName[i] for i in [sharps[q] for q in range(ks)]]
            ksFlat[trkNo] = set()
        else:
            ksFlat[trkNo] = [keyName[i] for i in [flats[q] for q in range(-ks)]]
            ksSharp[trkNo] = set()
    def parseClefOne(elem:Clef, lineNo:int):        
        nonlocal currentClef
        currentClef[lineNo%numTrack] = elem.type
        return elem.type
    def getClefFromType(clefType:int):
        clf = clef.TrebleClef()
        if clefType == 1:
            clf = clef.TrebleClef()
        elif clefType == 0:
            clf = clef.AltoClef()
        elif clefType == -1: #-1
            clf = clef.BassClef()
        elif clefType == -2:
            clf = clef.TenorClef()
        return clf

    def parseClef(elem:Clef, lineNo:int):
        nonlocal currentClef
        if elem.type == currentClef[lineNo%numTrack]:
            return None
        clf = None
        if elem.type == 1:
            clf = clef.TrebleClef()
        elif elem.type == 0:
            clf = clef.AltoClef()
        elif elem.type == -1: #-1
            clf = clef.BassClef()
        elif elem.type == -2:
            clf = clef.TenorClef()
        currentClef[lineNo%numTrack] = elem.type
        return clf
    def parseAccidentals(elem:Accidentals, lineNo:int):
        if not elem.isKeySignature: # it's accidentals
            return None
        ys = elem.shrinkYs
        if elem.ksKeySop is None:
            keyNum = np.sum(rr[ys[0]:ys[1],(lineNo+1)%4])/((ys[1]-ys[0]))*2-25
            elem.ksKeySop= keyNum
        return elem
    def calculatePitch(inputPitch:str):
        outputNum = keyCharFlat.index(inputPitch[0])+12*(int(inputPitch[-1])+1)
        if len(inputPitch)==3:
            if inputPitch[1] == '#':
                outputNum+=1
            elif inputPitch[1] == '-' :
                outputNum-=1
        return outputNum
    def pitchListShift(inputPitchList:List[str], shift: int, sharp = True): # shift: how many half notes
        returnList = []
        for pitch in inputPitchList:
            pitchNum = calculatePitch(pitch)
            pitchNum = pitchNum + shift
            returnList.append(pitchNum)
        referenceList = keyCharFlat
        if sharp:
            referenceList = keyCharSharp
        for idx,pitchNo in enumerate(returnList):
            clefNum = pitchNo//12-1
            pitchName = referenceList[pitchNo%12]
            returnList[idx] = pitchName + str(clefNum)
        return returnList
    def parseNoteGroupShift(elem:NoteGroup, flatSet, sharpSet, naturalSet, currClef, trkShift, flatSharp, linNo):
        # trkShift: 0 if original: -2, new: -2 (two flats), 1 if original: -2, new: -1
        # flatSharp: how many flats/sharp, -2: two flats 
        # flatSharp = 2, trkShift = 0: (-2), flatSharp = 2, trkShift = 1: (-1)
        keyLst = []
        currLength = 1
        regNoteCount = 0
        def setRemove(currSet, currElm):
            if currElm in currSet:
                currSet.remove(currElm)
            return currSet
        for stem in elem.noteStemList:
            if not stem.isOrnament:
                regNoteCount+=1
            else:
                continue
            actualPitch = stem.pitchSoprano
            if currClef == 0: #viola
                actualPitch -= 6
            elif currClef == -1: #cello
                actualPitch -= 12 
            elif currClef == -2: #tenor
                actualPitch -= 8
            currKeyName = keyName[(actualPitch-1)%7]
            currKey = currKeyName+str((actualPitch-1)//7+5)
            if stem.accidentals is not None: # it has keySignature
                sharpSet = setRemove(sharpSet, currKey)
                naturalSet = setRemove(naturalSet, currKey)
                flatSet = setRemove(flatSet, currKey)
                if stem.accidentals == -1 or currKey in flatSet:
                    flatSet.add(currKey)
                    currKey = currKey[0]+'-'+currKey[1]
                elif stem.accidentals == 1 or currKey in sharpSet:
                    sharpSet.add(currKey)
                    currKey = currKey[0]+'#'+currKey[1]
                else: # stem.accidentals == 0:
                    naturalSet.add(currKey)
            else:
                if currKey in naturalList:
                    currKey = currKey
                elif currKey in flatSet or currKeyName in ksFlat[linNo]:
                    currKey = currKey[0]+'-'+currKey[1]
                elif currKey in sharpSet or currKeyName in ksSharp[linNo]:
                    currKey = currKey[0]+'#'+currKey[1]
            # llen = float(Fraction(class_names[stem.rhythm]))
            # if stem.hasdot:
            #     llen = llen*1.5
            # if llen<currLength:
            #     currLength = llen
            keyLst.append(currKey)
        currLength = float(elem.tunedLength)
        if currLength == 0: # account for tuned to 0
            return flatSet, sharpSet, None
        keyShiftedList = keyLst
        trksOriginal[linNo].append(keyLst)
        keyShiftedList = pitchListShift(keyLst, (-trkShift*7)%12, flatSharp>0)
        trksShifted[linNo].append(keyShiftedList)
        if len(keyShiftedList) > 1:
            return flatSet, sharpSet, chord.Chord(keyShiftedList, quarterLength = currLength*4)
        elif len(keyShiftedList) == 1:
            return flatSet, sharpSet, note.Note(keyShiftedList[0], quarterLength=currLength*4)
        else:
            return flatSet, sharpSet, None
        # actualPitch, 0: B4, 1: C5, 2: D5
    
    def parseNoteGroup(elem:NoteGroup, flatSet, sharpSet, naturalSet, currClef, linNo, trkShift = 0):
        keyLst = []
        currLength = 1
        regNoteCount = 0
        def setRemove(currSet, currElm):
            if currElm in currSet:
                currSet.remove(currElm)
            return currSet
        for stem in elem.noteStemList:
            if not stem.isOrnament:
                regNoteCount+=1
            else:
                continue
            actualPitch = stem.pitchSoprano
            if currClef == 0: #viola
                actualPitch -= 6
            elif currClef == -1: #cello
                actualPitch -= 12 
            elif currClef == -2: #tenor
                actualPitch -= 8
            currKeyName = keyName[(actualPitch-1)%7]
            currKey = currKeyName+str((actualPitch-1)//7+5)
            if stem.accidentals is not None: # it has keySignature
                sharpSet = setRemove(sharpSet, currKey)
                naturalSet = setRemove(naturalSet, currKey)
                flatSet = setRemove(flatSet, currKey)
                if stem.accidentals == -1:
                    flatSet.add(currKey)
                    currKey = currKey[0]+'-'+currKey[1]
                elif stem.accidentals == 1 or currKey in sharpSet:
                    sharpSet.add(currKey)
                    currKey = currKey[0]+'#'+currKey[1]
                else: # stem.accidentals == 0:
                    naturalSet.add(currKey)
            else:
                if currKey in naturalList:
                    currKey = currKey
                elif currKey in flatSet or currKeyName in ksFlat[linNo]:
                    currKey = currKey[0]+'-'+currKey[1]
                elif currKey in sharpSet or currKeyName in ksSharp[linNo]:
                    currKey = currKey[0]+'#'+currKey[1]
            # llen = float(Fraction(class_names[stem.rhythm]))
            # if stem.hasdot:
            #     llen = llen*1.5
            # if llen<currLength:
            #     currLength = llen
            keyLst.append(currKey)
        currLength = float(elem.tunedLength)
        if currLength == 0: # account for tuned to 0
            return flatSet, sharpSet, None
        if len(keyLst) > 1:
            return flatSet, sharpSet, chord.Chord(keyLst, quarterLength = currLength*4)
        elif len(keyLst) == 1:
            return flatSet, sharpSet, note.Note(keyLst[0], quarterLength=currLength*4)
        else:
            return flatSet, sharpSet, None # ornament
        # actualPitch, 0: B4, 1: C5, 2: D5
    def parseRest(elem:Rest):
        restLength = float(elem.tunedLength)
        if restLength>0:
            # restLength = float(Fraction(restClassNames[elem.rhythm]))
            # if elem.hasdot:
            #     restLength = restLength*1.5
            return note.Rest(quarterLength=restLength*4)
        else:
            pass
            # print("setting rest to full")
            # return note.Rest(quarterLength=4)
    def assignKsCurrClef(accList:List[Accidentals], currentClef:int):
        pitchShift = 0
        if currentClef == 0:
            pitchShift -= 6
        elif currentClef== -1:
            pitchShift -= 12
        elif currentClef== -2:
            pitchShift -= 8
        i = 0
        while i<len(accList):
            actualKey = (accList[i].ksKeySop+pitchShift-1)%7
            if accList[i].shift == 1 and abs(actualKey - sharps[0]) <= 1:
                break
            if accList[i].shift == -1 and abs(actualKey - flats[0]) <= 1:
                break
            i+=1
        if i >= len(accList):
            return None
        totalLength = len(accList)-i
        pitchList = [(accList[r].ksKeySop+pitchShift-1)%7 for r in range(i, len(accList))]
        if not False in [abs(pitchList[k] - sharps[k%7])<1.2 or pitchList[k]-sharps[k%7]>5.8 for k in range(totalLength)]:
            return totalLength
        elif not False in [abs(pitchList[k] - flats[k%7])<1.2 or pitchList[k]-flats[k%7]>5.8 for k in range(totalLength)]:
            return -totalLength
        else:
            return None
    def assignKS(accList:List[Accidentals]):
        pitchShift = 0
        if currentClef[lineNo%numTrack] == 0:
            pitchShift -= 6
        elif currentClef[lineNo%numTrack] == -1:
            pitchShift -= 12
        elif currentClef[lineNo%numTrack] == -2:
            pitchShift -= 8
        i = 0
        while i<len(accList):
            actualKey = (accList[i].ksKeySop+pitchShift-1)%7
            if accList[i].shift == 1 and abs(actualKey - sharps[0]) <= 1:
                break
            if accList[i].shift == -1 and abs(actualKey - flats[0]) <= 1:
                break
            i+=1
        if i >= len(accList):
            return None
        totalLength = len(accList)-i
        pitchList = [(accList[r].ksKeySop+pitchShift-1)%7 for r in range(i, len(accList))]
        if not False in [abs(pitchList[k] - sharps[k%7])<1.2 or pitchList[k]-sharps[k%7]>5.8 for k in range(totalLength)]:
            return totalLength
        elif not False in [abs(pitchList[k] - flats[k%7])<1.2 or pitchList[k]-flats[k%7]>5.8 for k in range(totalLength)]:
            return -totalLength
        else:
            return None
    numBars = len(barList[0])
    ksBarMat = np.ones((numTrack, numBars))*np.inf
    clefMat = np.ones((numTrack, numBars))*np.inf 
    currentClef = [None]*numTrack
    for currBarNumber in range(numBars):
        barNumber = currBarNumber+1
        for lineNo in range(len(barList)):
            if len(CLEF_OPTIONS[lineNo]) == 1:
                clefMat[lineNo, currBarNumber] = CLEF_OPTIONS[lineNo][0]
            else:
                currBar = barList[lineNo][currBarNumber]
                for elem in currBar.elementList:
                    if type(elem) == Clef:
                        newClef = parseClefOne(elem, lineNo)
                        if newClef in CLEF_OPTIONS[lineNo]:
                            clefMat[lineNo, currBarNumber] = newClef
                if clefMat[lineNo, currBarNumber] == np.inf:
                    if currBarNumber == 0:
                        clefMat[lineNo, currBarNumber] = CLEF_OPTIONS[lineNo][0]
                    else:
                        clefMat[lineNo, currBarNumber] = clefMat[lineNo, currBarNumber-1]
    global DEBUGIMG
    beamMapRefIndex = -1
    for currBarNumber in range(numBars):
        if beamMapRefList is not None:
            if beamMapRefList[currBarNumber] != beamMapRefIndex:
                beamMapRefIndex = beamMapRefList[currBarNumber]
                beamMapImg = beamMapList[beamMapRefIndex]
                bb,gg,rr = cv2.split(beamMapImg)
        barNumber = currBarNumber+1
        for lineNo in range(len(barList)):
            currBar = barList[lineNo][currBarNumber]
            accumAcc: List[Accidentals] = [] # will have the list of accidentals to process    
            for elem in currBar.elementList:
                if type(elem) == Accidentals:
                    if lineNoList is not None:
                        currAcc = parseAccidentals(elem,lineNoList[currBarNumber]) # return none if it's accidentals
                    else:
                        currAcc = parseAccidentals(elem, lineNo)
                    if currAcc is None:
                        if debugXMLImg is not None:
                            x0,y0,x1,y1 = elem.boundingBox
                            debugXMLImg = cv2.rectangle(debugXMLImg, (x0,y0),(x1,y1),elem.getColor(), 1, cv2.LINE_AA)
                            DEBUGIMG = cv2.rectangle(DEBUGIMG, (x0,y0),(x1,y1),elem.getColor(), 1, cv2.LINE_AA)
                        if len(accumAcc)>0:
                            ksCurr = assignKsCurrClef(accumAcc, clefMat[lineNo, currBarNumber])
                            if ksCurr is None:
                                for ac in accumAcc:
                                    x0,y0,x1,y1 = ac.boundingBox
                                    debugXMLImg = cv2.circle(debugXMLImg, ((x0+x1)//2, (y0+y1)//2), (x1-x0)//2, ac.getColor(), 3, cv2.LINE_AA)
                                    DEBUGIMG = cv2.circle(DEBUGIMG, ((x0+x1)//2, (y0+y1)//2), (x1-x0)//2, ac.getColor(), 3, cv2.LINE_AA)
                                if barNumber == 1:
                                    modShift = statistics.mode([a.shift for a in accumAcc])
                                    currentKs[lineNo%numTrack] = modShift*len(accumAcc)
                                    ksBarMat[lineNo, currBarNumber] = modShift*len(accumAcc)    
                                    debugXMLImg = cv2.putText(debugXMLImg,str(modShift*len(accumAcc)), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ac.getColor(), 2, cv2.LINE_AA)
                                    DEBUGIMG = cv2.putText(DEBUGIMG,str(modShift*len(accumAcc)), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ac.getColor(), 2, cv2.LINE_AA)
                            else:
                                for ac in accumAcc:
                                    x0,y0,x1,y1 = ac.boundingBox
                                    debugXMLImg = cv2.rectangle(debugXMLImg, (x0,y0),(x1,y1),ac.getColor(), 3, cv2.LINE_AA)
                                    DEBUGIMG = cv2.rectangle(DEBUGIMG, (x0,y0),(x1,y1),ac.getColor(), 3, cv2.LINE_AA)
                                debugXMLImg = cv2.putText(debugXMLImg,str(ksCurr), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ac.getColor(), 2, cv2.LINE_AA)
                                DEBUGIMG = cv2.putText(DEBUGIMG,str(ksCurr), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ac.getColor(), 2, cv2.LINE_AA)
                            if ksCurr is not None and ksCurr != currentKs[lineNo%numTrack]:
                                currentKs[lineNo%numTrack] = ksCurr
                                ksBarMat[lineNo, currBarNumber] = ksCurr
                    elif currAcc.endKeySignature:
                        accumAcc.append(currAcc)
                        ksCurr = assignKsCurrClef(accumAcc, clefMat[lineNo, currBarNumber])
                        if ksCurr is None:
                            for ac in accumAcc:
                                x0,y0,x1,y1 = ac.boundingBox
                                debugXMLImg = cv2.circle(debugXMLImg, ((x0+x1)//2, (y0+y1)//2), (x1-x0)//2, ac.getColor(), 3, cv2.LINE_AA)
                                DEBUGIMG = cv2.circle(DEBUGIMG, ((x0+x1)//2, (y0+y1)//2), (x1-x0)//2, ac.getColor(), 3, cv2.LINE_AA)
                            if barNumber == 1:
                                modShift = statistics.mode([a.shift for a in accumAcc])
                                currentKs[lineNo%numTrack] = modShift*len(accumAcc)
                                ksBarMat[lineNo, currBarNumber] = modShift*len(accumAcc)
                                debugXMLImg = cv2.putText(debugXMLImg,str(modShift*len(accumAcc)), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ac.getColor(), 2, cv2.LINE_AA)
                                DEBUGIMG = cv2.putText(DEBUGIMG,str(modShift*len(accumAcc)), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ac.getColor(), 2, cv2.LINE_AA)
                        else:
                            for ac in accumAcc:
                                x0,y0,x1,y1 = ac.boundingBox
                                debugXMLImg = cv2.rectangle(debugXMLImg, (x0,y0),(x1,y1),ac.getColor(), 3, cv2.LINE_AA)
                                DEBUGIMG = cv2.rectangle(DEBUGIMG, (x0,y0),(x1,y1),ac.getColor(), 3, cv2.LINE_AA)
                            debugXMLImg = cv2.putText(debugXMLImg,str(ksCurr), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ac.getColor(), 2, cv2.LINE_AA)
                            DEBUGIMG = cv2.putText(DEBUGIMG,str(ksCurr), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ac.getColor(), 2, cv2.LINE_AA)
                        if ksCurr is not None and ksCurr != currentKs[lineNo%numTrack]:
                            currentKs[lineNo%numTrack] = ksCurr
                            ksBarMat[lineNo, currBarNumber] = ksCurr
                    else:
                        accumAcc.append(currAcc)
            if len(accumAcc)>0:
                ksCurr = assignKsCurrClef(accumAcc, clefMat[lineNo, currBarNumber])
                if ksCurr is not None:
                    currentKs[lineNo%numTrack] = ksCurr
                    ksBarMat[lineNo, currBarNumber] = ksCurr
            if ksBarMat[lineNo, currBarNumber] == np.inf:
                if currBarNumber in barsBreakPoints:
                    ksBarMat[lineNo, currBarNumber] = 0
                else:
                    ksBarMat[lineNo, currBarNumber] = ksBarMat[lineNo, currBarNumber-1]
    kk = ksBarMat.copy()
    for idx, trShift in enumerate(TRACK_SHIFT):
        kk[idx,:] = kk[idx,:]-trShift     
    ksBarMatNeutral = np.zeros_like(ksBarMat) # the "standard" orchestra clef we want
    ksBarMatNeutral[:,:] = stats.mode(kk, axis=0)[0]
    ksBarMatNew = np.zeros_like(ksBarMat) 
    for idx, trShift in enumerate(TRACK_SHIFT):
        ksBarMatNew[idx,:] = ksBarMatNeutral[idx,:]+trShift 
    ksBarMatNeutral = ksBarMatNeutral.astype(int)    
    ksBarMatNew = ksBarMatNew.astype(int)
    clefMat = clefMat.astype(int)
    tssList = [b.ts for b in barList[0]]
    for currBarNumber in range(numBars):
        barNumber = currBarNumber+1
        for lineNo in range(len(barList)):
            currBar = barList[lineNo][currBarNumber]
            measure = stream.Measure(number=barNumber)
            flatList = set() #put here because reset in measures
            sharpList = set()
            naturalList = set()
            accumAcc = [] # will have the list of accidentals to process
            clefType = clefMat[lineNo, currBarNumber]
            ks = ksBarMatNew[lineNo, currBarNumber]
            if currBarNumber == 0:
                measure.append(meter.TimeSignature(f'{currBar.ts[0]}/{currBar.ts[1]}'))
            elif tssList[currBarNumber]!=tssList[currBarNumber-1]:
                measure.append(meter.TimeSignature(f'{currBar.ts[0]}/{currBar.ts[1]}'))
            if currBarNumber == 0:
                measure.append(getClefFromType(clefType))
                setKS(ks, lineNo)
                measure.append(key.KeySignature(ks))
            elif clefType != clefMat[lineNo, currBarNumber-1]:
                measure.append(getClefFromType(clefType))
            elif ks != ksBarMatNew[lineNo, currBarNumber-1]:
                setKS(ks, lineNo)
                measure.append(key.KeySignature(ks))
            if currBar.getTunedTotalBeat() == 0: 
                measure.append(note.Rest(quarterLength = currBar.ts[0]/currBar.ts[1]*4))
            else:
                for elem in currBar.elementList:
                    if type(elem) == NoteGroup:
                        flatList, sharpList, currNote = parseNoteGroup(elem, 
                                                                    flatList, 
                                                                    sharpList, 
                                                                    naturalList, 
                                                                    clefMat[lineNo, currBarNumber],
                                                                    lineNo,
                                                                    trkShift=0)
                        if currNote is not None:
                            measure.append(currNote)
                        else:
                            print()
                    elif type(elem) == Rest:
                        currRest = parseRest(elem)
                        if currRest is not None:
                            measure.append(currRest)
            part[lineNo%numTrack].append(measure)
    for p in part:
        score.append(p)
    # Part 2: with shift
    ksSharp = [set() for _ in range(numTrack)]
    ksFlat = [set() for _ in range(numTrack)] 
    for currBarNumber in range(numBars): # for the new one shifted
        barNumber = currBarNumber+1
        for lineNo in range(len(barList)):
            currBar = barList[lineNo][currBarNumber]
            measure = stream.Measure(number=barNumber)
            # barNumber = barNumber+1
            flatList = set() #put here because reset in measures
            sharpList = set()
            naturalList = set()
            accumAcc = [] # will have the list of accidentals to process
            clefType = clefMat[lineNo, currBarNumber]
            ks = ksBarMatNeutral[lineNo, currBarNumber]
            if currBarNumber == 0:
                measure.append(getClefFromType(clefType))
                setKSNeutral(ks, lineNo, TRACK_SHIFT[lineNo])
                measure.append(key.KeySignature(ks))
            elif clefType != clefMat[lineNo, currBarNumber-1]:
                measure.append(getClefFromType(clefType))
            elif ks != ksBarMatNeutral[lineNo, currBarNumber-1]:
                setKSNeutral(ks, lineNo, TRACK_SHIFT[lineNo])
                measure.append(key.KeySignature(ks))
            if currBar.getTunedTotalBeat() == 0: 
                measure.append(note.Rest(quarterLength = currBar.ts[0]/currBar.ts[1]*4))
            for elem in currBar.elementList:
                if type(elem) == NoteGroup:
                    flatList, sharpList, currNote = parseNoteGroupShift(elem, 
                                                                   flatList, 
                                                                   sharpList, 
                                                                   naturalList, 
                                                                   clefMat[lineNo, currBarNumber],
                                                                   TRACK_SHIFT[lineNo],
                                                                   ksBarMatNeutral[0,currBarNumber],
                                                                   lineNo)
                    if currNote is not None:
                        measure.append(currNote)
                    else:
                        print()
                elif type(elem) == Rest:
                    currRest = parseRest(elem)
                    if currRest is not None:
                        measure.append(currRest)
            part2[lineNo%numTrack].append(measure)
    for p in part2:
        score2.append(p)
    return score, score2, {'ksAssigned':debugXMLImg}

def saveBarToCsv(imgName:str, barList:List[List[Bar]], desiredLength=1):
    retBarLst = []
    for lineId, oneLine in enumerate(barList):
        for barId, bar in enumerate(oneLine):
            if bar.getTotalBeat() != desiredLength:
                currBar = dict(sheetName=imgName, 
                               lineNo=lineId, 
                               barNo=barId, 
                               targetLength=desiredLength,
                               elms=[])
                for elm in bar.elementList:
                    if type(elm) == NoteGroup:
                        groupId = 0
                        if elm.noteChunkId is not None:
                            groupId = elm.noteChunkId
                        beamLength = (-1,-1)
                        beamEnd = [-1,-1] # for grouped notes
                        length = elm.getMinLength()
                        if len(elm.noteStemList) == 1:
                            currStm = elm.noteStemList[0]
                            if currStm.hasdot:
                                length = length*1.5
                            beamLength = [int(a) for a in currStm.getBeamHeights()]
                            if currStm.isup:
                                beamEnd = [currStm.getX(), int(min(currStm.getY0Y1()))]
                            else:
                                beamEnd = [currStm.getX(), int(max(currStm.getY0Y1()))]
                        newElm = dict(length=str(Fraction(length)),
                                        groupId = groupId,
                                        isRest = False,
                                        numNotes = len(elm.noteStemList),
                                        beamLength = beamLength,
                                        beamEnd=beamEnd)
                        currBar['elms'].append(newElm)
                    elif type(elm) == Rest:
                        newElm = dict(length=str(elm.getLengthFraction()*1.5 if elm.hasdot else elm.getLengthFraction()),
                                        groupId = -1,
                                        isRest = True,
                                        numNotes = 0,
                                        beamLength = (-1,-1),
                                        beamEnd=[-1,-1])
                        currBar['elms'].append(newElm)
                retBarLst.append(currBar)
    with open(f"debugBars/{imgName}.json", "w") as file:
        json.dump(retBarLst, file, indent=4) 

def tuneBarList(barList:List[List[Bar]], numBarsPerLine:List[int]):
    assert len(np.unique([len(barr) for barr in barList])) == 1
    emptyBarsIndexs = np.where(np.sum(np.array([[len(b.elementList) for b in oneTrack] for oneTrack in barList]),axis=0)==0)[0].tolist()
    emptyBarsIndexs.reverse()
    for emptyBarsIndex in emptyBarsIndexs:
        for i in range(NUM_TRACK):
            del barList[i][emptyBarsIndex]
        qq = 0
        while emptyBarsIndex+1>sum(numBarsPerLine[0:qq]):
            qq+=1
        numBarsPerLine[qq-1]-=1
    # delete all elements that's ornament
    for trackId, oneTrack in enumerate(barList):
        for barId, bar in enumerate(oneTrack):
            bar.elementList = [
                elem for elem in bar.elementList
                if not (isinstance(elem, NoteGroup) and all(s.isOrnament for s in elem.noteStemList))
            ]
    # now all bars are suppose to be valid
    for trackId, oneTrack in enumerate(barList):
        for barId, bar in enumerate(oneTrack):
            origLengths = bar.getRhythmList()
            newLengths = [None]*len(origLengths)
            if bar.getTotalBeat() == Fraction(bar.ts[0],bar.ts[1]):
                newLengths = [None]*len(origLengths)
            elif bar.getTotalBeat() ==  0:
                continue # will be later added a rest
            else:
                newLengths = tuneBar(bar.getRestNg(), bar.ts)
            reassignedLengths = [origLengths[i] if newLengths[i] is None else newLengths[i] for i in range(len(newLengths))]
            bar.reassignLength(origLengths, newLengths)
            if bar.getTunedTotalBeat() != Fraction(bar.ts[0], bar.ts[1]):
                print("bar doesn't have right length")

    for lineii, oneTrack in enumerate(barList):
        for barii, bb in enumerate(oneTrack):
            if bb.getTunedTotalBeat() != Fraction(bb.ts[0],bb.ts[1]):
                print(f"bar at {lineii},{barii} has total beat {bb.getTotalBeat()}")
    assert sum(numBarsPerLine) == len(barList[0])
    barBreakPoints = [0] + list(itertools.accumulate(numBarsPerLine))
    return barBreakPoints

def extendNotegroupsToStaff(noteGroupMap:np.ndarray, 
                            noteGroupVerticallyMerged:List[NoteGroup],
                            staffObjList:List[Staff],
                            beamMapImg:np.ndarray):
    bb,gg,rr = cv2.split(beamMapImg)
    ngZero = np.zeros_like(noteGroupMap)
    for sf in staffObjList:
        ngZero[sf.ys[0]:sf.ys[-1], sf.left:sf.right] = 1
    for idx, ng in enumerate(noteGroupVerticallyMerged):
        if ng is None:
            continue
        x0,y0,x1,y1 = ng.boundingBox
        if np.sum(ngZero[y0:y1,x0:x1])==0:
            clefNo = round(np.mean(rr[y0:y1,4]))
            if len(np.unique(noteGroupMap[y0:y1,x0:x1])) == 1:
                noteNo =np.unique(noteGroupMap[y0:y1,x0:x1])[0]
                if y0>staffObjList[clefNo-1].ys[2]: # is at the bottom
                    y00 = staffObjList[clefNo-1].ys[-2]
                    y11 = y1
                else:
                    y00 = y0
                    y11 = staffObjList[clefNo-1].ys[1]            
                gg[y00:y11,x0:x1] = 1
                noteGroupMap[y00:y11,x0:x1]=noteNo
                rr[y00:y11,x0:x1] = clefNo
                ng.boundingBox = (x0,y00,x1,y11)

    return cv2.merge([bb,gg,rr]), noteGroupMap

def getAccidentalsChanges(barList:List[List[Bar]]):
    # barList: [[track 1's bars], [track 2's bars], ...]
    numTracks = len(barList)
    numBars = len(barList[0])
    accTracks = [[0]*numBars for _ in range(numTracks)]
    clefTracks = [[None]*numBars for _ in range(numTracks)]
    imgCircleAcc = image.copy()
    clrs = [(0,0,255),(0,255,0),(255,255,0)]
    for lineNo,currBarList in enumerate(barList):
        for barNo,currBar in enumerate(currBarList):
            for elem in currBar.elementList:
                if type(elem) == Clef:
                    # 1: treble, 0: alto, -1: bass, -2: tenor
                    clefTracks[lineNo][barNo] = elem.type
                    print(f"at line {lineNo}, bar {barNo}, element has type {elem.type}")
            if clefTracks[lineNo][barNo] is None and barNo>0:
                clefTracks[lineNo][barNo] = clefTracks[lineNo][barNo-1]
    #  TODO: fill in if there's none in clefTracks
    if [True] in [[None in f] for f in clefTracks]:
        print("need to reassign!")
    for lineNo,currBarList in enumerate(barList):
        for barNo,currBar in enumerate(currBarList):
            for elem in currBar.elementList:
                if type(elem) == Accidentals and elem.isKeySignature:
                    x0,y0,x1,y1 = elem.boundingBox
                    currClef = clefTracks[lineNo][barNo]
                    # ys = elem.shrinkYs
                    # keyNum = np.sum(rr[ys[0]:ys[1],(lineNo+1)%4])/((ys[1]-ys[0]))*2-25
                    # elem.ksKeySop = keyNum
                    imgCircleAcc = cv2.rectangle(imgCircleAcc,(x0,y0),(x1,y1),clrs[elem.shift],3, cv2.LINE_AA)

def createMask(barList:List[List[Bar]], barRanges:List[List[Tuple[int,int]]], staffObjList:List[Staff], image:np.ndarray,beamMapImg:np.ndarray):
    maskImg = image.copy()
    brWidth = 2 # VARIABLE: 1 third of the size of mask
    barlineCenters = []
    tsBounding1 = image.copy()
    tsBounding2 = image.copy()

    for lineIdx,brList in enumerate(barRanges):
        yStart = staffObjList[lineIdx*NUM_TRACK].ys[0]
        yEnd = staffObjList[(lineIdx+1)*NUM_TRACK-1].ys[-1]
        brCenters = [br[0] for br in brList]+[brList[-1][-1]]
        barlineCenters.append(brCenters)
        for br in brCenters:
            maskImg = cv2.rectangle(maskImg, (br-brWidth, yStart),(br+2*brWidth, yEnd), (255,255,255), lineType=cv2.LINE_AA, thickness=-1)
            tsBounding2 = cv2.rectangle(tsBounding2, (br-brWidth, yStart),(br+2*brWidth, yEnd), (0,255,0), lineType=cv2.LINE_AA, thickness=-1)
    for tracks in barList:
        for bar in tracks:
            for elem in bar.elementList:
                if type(elem) == Accidentals:
                    x0,y0,x1,y1 = elem.boundingBox
                    maskImg = cv2.rectangle(maskImg, (x0,y0),(x1,y1), (255,255,255), lineType=cv2.LINE_AA, thickness=-1)
                elif type(elem) == NoteGroup:
                    x0,y0,x1,y1 = elem.boundingBox
                    maskImg = cv2.rectangle(maskImg, (x0,y0),(x1,y1), (255,255,255), lineType=cv2.LINE_AA, thickness=-1)
                    for stm in elem.noteStemList:
                        if stm.accidentalBox is not None:
                            x0,y0,x1,y1 = stm.accidentalBox
                            maskImg = cv2.rectangle(maskImg, (x0,y0),(x1,y1), (255,255,255), lineType=cv2.LINE_AA, thickness=-1)
                elif type(elem) == Rest:
                    x0,y0,x1,y1 = elem.boundingBox
                    maskImg = cv2.rectangle(maskImg, (x0,y0),(x1,y1), (255,255,255), lineType=cv2.LINE_AA, thickness=-1)
                elif type(elem) == Clef:
                    x0,y0,x1,y1 = elem.boundingBox
                    maskImg = cv2.rectangle(maskImg, (x0,y0),(x1,y1), (255,255,255), lineType=cv2.LINE_AA, thickness=-1)
    beamMapBW = beamMapImg[:,:,0]
    _, bmINV = cv2.threshold(beamMapBW, 127,1, cv2.THRESH_BINARY_INV)
    maskImgBW = cv2.cvtColor(maskImg, cv2.COLOR_BGR2GRAY)      
    _, maskImgINV = cv2.threshold(maskImgBW, 127, 1, cv2.THRESH_BINARY_INV)
    imgBW = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    maskImgINV = maskImgINV*bmINV
    maskWhole = np.zeros_like(maskImgBW)
    yPrev = 0
    for sfId,sf in enumerate(staffObjList):
        y0 = sf.ys[0]
        y1 = sf.ys[-1]
        maskWhole[yPrev:y0,:] = 0
        x0 = sf.left
        x1 = sf.right
        currLine = maskImgINV[y0:y1,:]
        h,w = currLine.shape
        sumCurr = np.sum(currLine,axis=0)
        val,cnts = np.unique(sumCurr, return_counts = True)
        sorted_indices = np.argsort(-cnts)
        minIdx = np.where(sorted_indices>=np.percentile(np.sum(currLine,axis=0),75))[0][0]
        thres = np.max(sorted_indices[:3])
        thres = max(thres, sorted_indices[minIdx])
        xPos = np.where(np.sum(currLine,axis=0)>thres)[0]
        mask = np.zeros(w, dtype=np.uint8)
        mask[xPos] = 1
        mask[:x0] = 0
        mask[x1:] = 0
        mask = np.tile(mask, (h,1))
        maskWhole[y0:y1,:] = mask
        yPrev = y1
    maskWhole[yPrev:,:] = 0
    retImgBlack = cv2.bitwise_not(imgBW)*maskWhole
    retImgWhite = cv2.bitwise_not(maskWhole*maskImgINV*255)
    contours, _ = cv2.findContours(retImgBlack, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rett = image.copy()
    whiteBG = np.zeros_like(imgBW) # put 1s on where there's residual stuff
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h>sf.get_yOne():
            rett = cv2.rectangle(rett, (x,y),(x+w,y+h),(0,0,255),-1,cv2.LINE_AA)
            whiteBG = cv2.rectangle(whiteBG, (x,y),(x+w,y+h),1,-1,cv2.LINE_AA)
    sfTrack = np.zeros((len(staffObjList),image.shape[1]))
    for sfId,sf in enumerate(staffObjList):
        y0 = sf.ys[0]
        y1 = sf.ys[-1]
        currLine = whiteBG[y0:y1,:]
        xLoc = np.max(currLine, axis=0)
        sfTrack[sfId,:] = xLoc
    windowWidth = sf.get_yOne() #VARIABLE: how much we want the sliding window to be (width)
    windowThres = 0.5
    boxList = [(0,0,0,0,-1)] # x0,y0,x1,y1,lineNo(starting 0)
    for linNo in range(len(staffObjList)//NUM_TRACK): # go through window and record
        currSFTrack = sfTrack[linNo*NUM_TRACK:(linNo+1)*(NUM_TRACK),:]
        for xLeft in range(0, image.shape[1]-windowWidth):
            if np.mean(currSFTrack[:,xLeft:xLeft+windowWidth])>windowThres:
                if xLeft<staffObjList[linNo*NUM_TRACK].left:
                    continue
                y0 = staffObjList[linNo*NUM_TRACK].ys[0]
                y1 = staffObjList[(linNo+1)*NUM_TRACK-1].ys[-1]
                if boxList[-1][4] == linNo and boxList[-1][2] == xLeft+windowWidth-1:
                    xx0,yy0,xx1,yy1,_ = boxList[-1]
                    boxList[-1] = (xx0,yy0,xx1+1,yy1,linNo)
                else:
                    boxList.append((xLeft, y0, xLeft+windowWidth, y1, linNo))
    boxListFiltered = boxList[1:].copy()
    currBrCenter = [0,-len(barlineCenters[0])]
    for box in boxList:
        x0,y0,x1,y1,currLine = box
        tsBounding1 = cv2.rectangle(tsBounding1, (x0,y0), (x1,y1), (0,0,255),3,cv2.LINE_AA)
        if currLine != currBrCenter[0]:
            currBrCenter = [currLine,-len(barlineCenters[currLine])]
        if currBrCenter[-1] >=0: # no overlap for sure
            continue
        while barlineCenters[currBrCenter[0]][currBrCenter[1]]<x0 and currBrCenter[-1]<0:
            currBrCenter[-1]+=1
        if barlineCenters[currBrCenter[0]][currBrCenter[1]]<x1:
            boxListFiltered.remove(box)
    for idx,box in enumerate(boxListFiltered):
        x0,y0,x1,y1,currLine = box
        cropImgLst = []
        for trkNo in range(NUM_TRACK):
            yss = staffObjList[currLine*NUM_TRACK+trkNo].ys
            crpImg = imgBW[yss[0]:yss[-1],x0:x1]
            cropImgLst.append(crpImg)
        matchScores = []
        for i in range(4):
            for j in range(i+1,4):
                result = cv2.matchTemplate(cropImgLst[i][:,0:min([c.shape[0] for c in cropImgLst])], cropImgLst[j], cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                matchScores.append(max_val)
        if min(matchScores)>0:
            tsBounding2 = cv2.rectangle(tsBounding2, (x0,y0), (x1,y1), (0,0,255),3,cv2.LINE_AA)
            for idx,crpImg in enumerate(cropImgLst):
                cv2.imwrite(f"ts_dataset/{img_name}_{x0}_{y0}_{idx}.jpg", crpImg)
                print(f"writing dataset to ts_dataset/{img_name}_{x0}_{y0}_{idx}.jpg")
    imwrite("TSboundingBox_.jpg", tsBounding1)
    imwrite("TSboundingBoxNoBr_.jpg", tsBounding2)
    imwrite("filteredTS_.jpg",rett) 
    return maskImg, boxList, boxListFiltered,{'tsboundingBox':tsBounding1,'tsboundingBoxNoBr':tsBounding2, 'filteredTS':rett}

    for box in boxList[1:]:
        x0,y0,x1,y1,currLine = box
        tsBounding1 = cv2.rectangle(tsBounding1, (x0,y0), (x1,y1), (0,0,255),3,cv2.LINE_AA)
        cropImgLst = []
        for trkNo in range(NUM_TRACK):
            yss = staffObjList[currLine*NUM_TRACK+trkNo].ys
            crpImg = imgBW[yss[0]:yss[-1],x0:x1]
            cropImgLst.append(crpImg)
        matchScores = []
        for i in range(4):
            for j in range(i+1,4):
                result = cv2.matchTemplate(cropImgLst[i], cropImgLst[j], cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                matchScores.append(max_val)
        if min(matchScores)>0:
            tsBounding2 = cv2.rectangle(tsBounding2, (x0,y0), (x1,y1), (0,0,255),3,cv2.LINE_AA)
            boxListFiltered.append(box)
            for idx,crpImg in enumerate(cropImgLst):
                cv2.imwrite(f"ts_dataset/{img_name}_{x0}_{y0}_{idx}.jpg", crpImg)
    imwrite("TSboundingBox_.jpg", tsBounding1)
    imwrite("TSboundingBoxNoBr_.jpg", tsBounding2)
    imwrite("filteredTS_.jpg",rett) 
    return maskImg, boxList, boxListFiltered,{'tsboundingBox':tsBounding1,'tsboundingBoxNoBr':tsBounding2, 'filteredTS':rett}
# pagenum starts with 1
def editBarTS(barList:List[List[Bar]], allChanges:List[dict], pageNum, numBarsPerLine:List[int]):
    prev = max([i for i, val in enumerate([allChanges[i]['page'] for i in range(len(allChanges))]) if val < pageNum]+[0]) 
    TIME_SIGNATURE = allChanges[prev]['time_signature']
    indexes = [i+1 for i, val in enumerate([allChanges[i]['page'] for i in range(1,len(allChanges))]) if val == pageNum]        
    if len(indexes) > 0:
        breakPoints = [0] + list(itertools.accumulate(numBarsPerLine))
        numBarsThisPage = len(barList[0]) # how many bars are in this page
        locBarNumChanges = [allChanges[ind]['loc'] for ind in indexes]
        barNumChanges = [breakPoints[l[0]]+l[1] for l in locBarNumChanges]+[numBarsThisPage*2] # [0,6, xx (too large)] if we have ts change in bar 0, 6
        tss = [TIME_SIGNATURE]+[allChanges[ind]['time_signature'] for ind in indexes] # [[orig],[3/4],[9/8]]
        # from 0 to barNumChanges[0] : TIME_SIGNATURE
        # from barNumChanges[0] to barNumChanges[1]: tss[0]
        for currTrack in barList:
            currTsIndex = 0 # tss[currTsIndex] is the one we want to assign
            TIME_SIGNATURE = tss[0]
            for barNum, bar in enumerate(currTrack):
                if barNum == barNumChanges[currTsIndex]:
                    currTsIndex += 1
                    TIME_SIGNATURE = tss[currTsIndex]
                bar.ts = TIME_SIGNATURE
    else:
        for currTrack in barList:
            for barNum, bar in enumerate(currTrack):
                bar.ts = TIME_SIGNATURE

if __name__ == '__main__':
    base_folder = 'string_dataset/pdf_data/'
    base_output_folder = 'string_dataset/output/'
    for piece_name in [f"mendelssohn1"]:
        # piece_name = 'beethoven2'
        OUTPUT_BASE_FOLDER = os.path.join(base_output_folder, piece_name)
        if not os.path.isdir(OUTPUT_BASE_FOLDER):
            os.mkdir(OUTPUT_BASE_FOLDER)
        if not os.path.isdir(f"{OUTPUT_BASE_FOLDER}/output"):
            os.mkdir(f"{OUTPUT_BASE_FOLDER}/output")
        piece_base_folder = os.path.join(base_folder, piece_name) # 'string_dataset/pdf_data/beethoven1'
        with open(f"{piece_base_folder}/{piece_name}.json") as f:
            config = json.load(f)
            NUM_TRACK = config['numTrack']
            TRACK_SHIFT = config['track_shift']
            CLEF_OPTIONS = config['clef_options']
            TIME_SIGNATURE = config['tsChange'][0]['time_signature']

        run_img_folder = os.path.join(piece_base_folder, 'imgs')
        if not os.path.isdir(run_img_folder):
            savePdf2Png(piece_base_folder,config['numPerPage'],config['rotate'])
        
        stemUpPth = "training/stemupImg32x32_best.pth"
        stemDownPth = "training/stemdownImg32x32_best.pth"
        stemUpModel = Single_Stem_Classifier(stemUpPth, 3)
        stemDownModel = Single_Stem_Classifier(stemDownPth, 3)

        img_name_list = [f'{piece_name}_{i}' for i in range(1,config['numPage']+1)]
        wholeBarList:List[List[Bar]] = [[] for _ in range(NUM_TRACK)]
        beamRefList: List[int] = [] # which beamMapImg should it refer to (which page was it in), start with 0
        allBeamMaps: List[np.ndarray] = []
        lineNoList: List[int] = []
        allBarBreakPoints: List[int] = []
        for imgIdx,img_name in enumerate(img_name_list):
            minBarheight = 8
            img_path = f"{run_img_folder}/{img_name}/{img_name}.png"
            base_path = f"{run_img_folder}/{img_name}/{img_name}"
            npy_path = f"{run_img_folder}/{img_name}/{img_name}.npy"
            pkl_path = f"{run_img_folder}/{img_name}/{img_name}_staffList.pkl"
            barList_path = f"{run_img_folder}/{img_name}/{img_name}_barlist.pkl"
            # beamMapImg_path = f"{run_img_folder}/{img_name}/{img_name}_beamMapInit.jpg"
            if not os.path.exists(npy_path):
                printt(f"missing npy for {img_name}")
                dataDictOrigin = runModel1(img_path,outputPath = base_path,dodewarp=False,save_npy=True)
            if not os.path.exists(pkl_path):
                printt(f"missing pkl for {img_name}")
                staffListOrigin = runModel2(npy_path, img_path, base_path)
            if os.path.exists(npy_path) and os.path.exists(pkl_path):
                printt(f"working on {img_path}")
            # if os.path.exists(beamMapImg_path): # beamMap for the rr in accidentals
            #     beamMapImg = cv2.imread(beamMapImg_path)
            #     with open(barList_path, "rb") as f:
            #         barLists = pickle.load(f)
            #     for tr in range(NUM_TRACK):
            #         wholeBarList[tr] += barLists[tr]
            #     beamRefList += [imgIdx]*len(barLists[0])
            #     lineNoList += [i for i in range(len(barLists[0]))]
            #     allBeamMaps.append(beamMapImg)
            #     continue
            # '''
            dataDict = np.load(npy_path,allow_pickle=True)
            dataDict = dataDict.tolist()
            resize_ratio = 2
            if img_name.startswith('packed_'):
                for k in dataDict.keys():
                    img = dataDict[k]
                    resized_image = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_NEAREST)
                    dataDict[k] = resized_image
            else:
                resize_ratio = 2
                for k in dataDict.keys():
                    img = dataDict[k]
                    resized_image = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_NEAREST)
                    dataDict[k] = resized_image

            img = np.ones_like(dataDict['image'])*255
            ys,xs = np.where(dataDict['stems_rests']>0)
            img[ys,xs] = (0,0,255)
            # outputImWrite(f'{img_name}_stemrests.jpg', img)
            imwrite(f'stemrests_.jpg', img)

            img = np.ones_like(dataDict['image'])*255
            ys,xs = np.where(dataDict['symbols']>0)
            img[ys,xs] = (255,0,0)
            # outputImWrite(f'{img_name}_symbols.jpg', img)
            imwrite(f'symbols_.jpg', img)

            # ori_img, binary_img, symbol = getStemBox(dataDict)
            bar_height, staffObjList = init_bar_height(dataDict, min_barheight=minBarheight)
            print(f'barheight: {bar_height}')
            print(bar_height)
            stepSize = int(bar_height/4)

            # get the beam image
            beamNoClefKeyWithStemRest, beamWithoutStemRest, beam_nostem = getBeamImage(dataDict, bar_height, staffObjList)
            # get the gradient beamMap and (dont filter noteheadInitial with longer beams cause might remove noteheads)
            beamMapImg, debugImages = generateBeamStaffImg(beam_nostem, staffObjList, bar_height,stepSize=stepSize)
            # get the inital black and white image of notehead (white: has note), it's b&w, depth=1
            # noteBwImage is the cleaned but not filtered note image, 
            
            # if os.path.exists(barList_path):
            #     with open(barList_path, "rb") as f:
            #         barLists = pickle.load(f)
            #     for tr in range(NUM_TRACK):
            #         wholeBarList[tr] += barLists[tr]
            #     beamRefList += [imgIdx]*len(barLists[0])
            #     cv2.imwrite(beamMapImg_path, beamMapImg)
            #     lineNoList += [i for i in range(len(barLists[0]))]
            #     allBeamMaps.append(beamMapImg)
            #     continue

            noteheadBoxesInitList, noteBwImageOrigin = getInitialNoteheadBoxList(dataDict, beam_nostem, beamMapImg, bar_height)

            # mask the note image with the notehead boxes
            maskedNoteBwImage = maskImage(noteheadBoxesInitList, noteBwImageOrigin)

            stem_init_list, image, debugImages= getStemList(dataDict, noteheadBoxesInitList, beamNoClefKeyWithStemRest)
            # {'notehead_drawimg':drawimg}
            for si in debugImages.keys():
                # outputImWrite(f'{img_name}_{si}.jpg', debugImages[si])
                imwrite(f'{si}.jpg', debugImages[si])
            stem_list_assigned_height, debugImages = assignStemLength(stem_init_list, image, beamMapImg, stepSize, bar_height)
            # {'notestem_drawimg': drawimg2}
            for si in debugImages.keys():
                # outputImWrite(f'{img_name}_{si}.jpg', debugImages[si])
                imwrite(f'{si}.jpg', debugImages[si])

            beamRemoveNotes = removeNotes(maskedNoteBwImage, beamNoClefKeyWithStemRest, beamMapImg)
            beam_img2, stem_list_assigned, beam_heights = assignBeamLengthBeamImg(stem_list_assigned_height, image, beamRemoveNotes, bar_height)
            # outputImWrite(f'{img_name}_knn_beam.jpg', beamRemoveNotes)
            # outputImWrite(f'{img_name}_knn_beamImg.jpg', beam_img2)
            imwrite(f'knn_beam.jpg', beamRemoveNotes)
            imwrite(f'knn_beamImg.jpg', beam_img2)

            noteGroupList,  beamMapImg, noteGroupStemMap, noteGroupMap, debugImages = knnRhythmAndDraw(stem_list_assigned, beam_heights, image, bar_height, beamMapImg, stemUpClassifier=stemUpModel, stemDownClassifier=stemDownModel,img_name=img_name)
            # {'knnbeams':imgrgb,'ngImg': ngImg, 'knnbeams2': knnbeams2}
            for si in ['ngImg','knnbeams2']:#debugImages.keys():
                # outputImWrite(f'{img_name}_notehead_Boxes.jpg', drawimg)
                outputImWrite(f'{img_name}_{si}.jpg', debugImages[si])
                imwrite(f'{si}.jpg',debugImages[si])
            
            ys,xs = np.where(noteGroupStemMap>0)
            zz = image.copy()
            zz[ys,xs,1:2] = 100
            # outputImWrite(f'{img_name}_noteGroupStemMap.jpg', zz)
            imwrite(f'noteGroupStemMap.jpg', zz)
            
            
            ys,xs = np.where(noteGroupMap>0)
            zz = image.copy()
            zz[ys,xs,1:2] = 100
            # outputImWrite(f'{img_name}_noteGroupStemMap.jpg', zz)
            imwrite(f'noteGroupMap.jpg', zz)

            colors = [(0,0,255),(0,255,0),(255,255,0), (0,120,230)]
            _,_,rr = cv2.split(beamMapImg)
            zz = image.copy()
            for i in range(len(staffObjList)):
                ys,xs = np.where(rr==(i+1))
                zz[ys,xs] = colors[i%4]
            imwrite('GroupMap.jpg',zz)
            # outputImWrite(f'{img_name}_GroupMap_1.jpg', zz)

            pth_path = r'training\rest_remain2x1_best.pth'
            num_classes = 5
            restClassifier = Rest_Classifier(pth_path, num_classes)
            restMap, restList, beamMapImg, debugImages = findRests(dataDict, beamMapImg, bar_height, restClassifier)
            #  {'RestClassification': imgrgb}
            # for si in debugImages.keys():
            #     outputImWrite(f'{img_name}_{si}.jpg', debugImages[si])
                # pass

            # beamMapImg: modified beamMap1 with vertically grouped ones
            noteGroupVerticallyMerged, noteGroupMapWithIssues, beamMapImg, debugImages = mergeVerticalNoteGroups(staffObjList,noteGroupList, noteGroupMap, beamMapImg, image,restMap, restList)
            
            noteGroupMap = np.zeros_like(noteGroupMapWithIssues)
            for i in range(len(noteGroupVerticallyMerged)):
                if noteGroupVerticallyMerged[i] is not None:
                    ng = noteGroupVerticallyMerged[i]
                    x0,y0,x1,y1 = ng.boundingBox
                    noteGroupMap[y0:y1, x0:x1] = i

            # {'vertically_merged':ngImg}
            for si in debugImages.keys():
                # outputImWrite(f'{img_name}_notehead_Boxes.jpg', drawimg)
                outputImWrite(f'{img_name}_{si}.jpg', debugImages[si])
                imwrite(f'{si}.jpg',debugImages[si])
            colors = [(0,0,255),(0,255,0),(255,255,0), (0,120,230)]
            _,_,rr = cv2.split(beamMapImg)
            zz = image.copy()
            for i in range(len(staffObjList)):
                ys,xs = np.where(rr==(i+1))
                zz[ys,xs] = colors[i%4]
            # outputImWrite(f'{img_name}_GroupMap_2.jpg', zz)
            imwrite(f'GroupMap2.jpg', zz)


            pth_path = 'training/best_model32x32_seg_dil4x4.pth'
            num_classes = 7
            sfnClassifier = Sfn_Clef_classifier(pth_path, num_classes)
            sfnClefList, sfnClefMap, debugImages = symbol_classification(dataDict, sfnClassifier, bar_height)
            # {'sfnClef':sfnClefImg}
            for si in debugImages.keys():
                outputImWrite(f'{img_name}_{si}.jpg', debugImages[si])
                imwrite(f'{si}.jpg',debugImages[si])
            
            # filter the sfn and clef that are overlapping with beam map
            sfnClefList, sfnClefMap, beamMapImg = filterSfnClefModifyBeamMap(sfnClefList, noteGroupMap, beamMapImg)
            bb,gg,_ = cv2.split(beamMapImg)
            _,zz = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
            zzmask = zz.copy()
            zzmask[:max(staffObjList[0].ys[0]-bar_height*2,0),:] = (255,255,255)
            zzmask[min(staffObjList[-1].ys[-1]+bar_height*2,zzmask.shape[0]):,:] = (255,255,255)
            for sf in staffObjList:
                xMaskOut = np.where(np.max(noteGroupMap[max(sf.ys[0]-2*bar_height, 0):min(sf.ys[-1]+2*bar_height, zzmask.shape[0]), :],0)>0)[0]
                zzmask[max(sf.ys[0]-2*bar_height, 0):min(sf.ys[-1]+2*bar_height,zzmask.shape[0]),xMaskOut] = (255,255,255)
            ys,xs = np.where(gg>0)
            zzmask[ys,xs] = (255,255,255)
            bb = cv2.dilate(bb, np.ones((bar_height//2, bar_height//2), dtype= np.uint8), iterations=1)
            ys,xs = np.where(bb>=255)
            zzmask[ys,xs] = (255,255,255)
            # zzmask = cv2.dilate(zzmask, np.ones((bar_height//3,1),dtype=np.uint8),iterations=1)
            # zzmask = cv2.erode(zzmask, np.ones((bar_height//3,bar_height//3),dtype=np.uint8),iterations=1)
            # cv2.imwrite('test.jpg',zzmask)
            imwrite('SfnClefNoteWhite.jpg', zzmask)
            # outputImWrite(f'{img_name}_SfnClefNoteWhite.jpg', zzmask)

            for i in range(1,4):
                ys,xs = np.where(gg==i)
                zz[ys,xs,(i-2)%3] = 100
                zz[ys,xs,(i-3)%3] = 0
            ys,xs = np.where(image[:,:,0]<128)
            zz[ys,xs,:] = (0,0,0)
            # outputImWrite(f'{img_name}_SfnClefNote.jpg', zz)
            imwrite(f'SfnClefNote.jpg', zz)


            staffEnhancedImg = enhanceStaff(image, staffObjList,bar_height)
            noteGroupPitchList, debugImages = assignPitch(staffEnhancedImg,noteGroupList, beamMapImg, bar_height)
            # {'pitch':pitchImg, 'pitchAndLine':pitchAndLineImg, 'pitchline': pitchLineImg}
            for si in debugImages.keys():
                # outputImWrite(f'{img_name}_{si}.jpg', debugImages[si])
                imwrite(f'{si}.jpg',debugImages[si])
            outputImWrite(f'{img_name}_pitchline.jpg', debugImages['pitchline'])



            imgMasked = getMaskedImageTimeSignature(image, beamMapImg, staffObjList, bar_height, img_name)
            # outputImWrite(f'{img_name}_imgMasked.jpg', imgMasked)
            # outputImWrite(f'{img_name}_tsContour.jpg', img2)

            CBounding = getCTimeSignature(image, imgMasked, staffObjList)
            outputImWrite(f'{img_name}_CBounding.jpg', CBounding)
            
            Bounding4 = getTimeSignature4(image, imgMasked, staffObjList)
            outputImWrite(f'{img_name}_4Bounding.jpg', Bounding4)
            noteChunkList, horizontalGroupImg = getNoteChunks(image, noteGroupMap, beamMapImg, noteGroupVerticallyMerged)
            # outputImWrite(f'{img_name}_beamContour.jpg',retimg)
            outputImWrite(f'{img_name}_horizontalGroupImg.jpg',horizontalGroupImg)

            beamMapImg, barlineboxes, barlineQuart = findBarlines(image, beamMapImg, staffObjList, noteGroupStemMap, thres=0.8) # VARIABLE thres
            outputImWrite(f'{img_name}_barlineQuart.jpg',barlineQuart)

            # barlineImages = filterBarlineBoxes(image, barlineboxes)
            # outputImWrite(f'{img_name}_barlineFiltered.jpg', barlineImages)

            zz = image.copy()
            bb,gg,_ = cv2.split(beamMapImg)
            colors = [(190,120,0),(255,50,0),(50,240,0),(0,120,230),(0,0,255)]
            for i in range(1,6):
                ys,xs = np.where(gg==i)
                zz[ys,xs] = colors[i%5]

            imwrite(f'SfnClefNoteBarlineRest.jpg', zz)

            colors = [(230,200,130),(230,130,100),(120,170,140),(235,176,113),(0,0,255)]
            zz = image.copy()
            for i in [1,2,3,5]:
                ys,xs = np.where(gg==i)
                zz[ys,xs] = colors[i%5]
            ys,xs = np.where(image[:,:,1]<128)
            zz[ys,xs] = (0,0,0)
            i = 4
            ys,xs = np.where(gg==i)
            zz[ys,xs] = colors[i%5]
            outputImWrite(f'{img_name}_DotPrevious.jpg', zz)
            imwrite(f'DotPrevious.jpg', zz)

            noteGroupVerticallyMerged, debugImages = assignDots(noteGroupVerticallyMerged,beamMapImg, image,bar_height,staffObjList)
            # {'dotBox':imgbgr}
            # for si in debugImages.keys():
            #     outputImWrite(f'{img_name}_{si}.jpg', debugImages[si])

            restList, debugImages = assignRestDots(restList, beamMapImg, noteGroupStemMap, image, bar_height, staffObjList)
            # {'dotRestBox':imgbgr}
            # for si in debugImages.keys():
            #     outputImWrite(f'{img_name}_{si}.jpg', debugImages[si])
            
            stemIdxMap, noteGroupVerticallyMerged, debugImages = assignSfnToNote(image, noteGroupMap, noteGroupVerticallyMerged,sfnClefMap,sfnClefList)
            # {'assignedAccidentals':accidentalsImg}
            for si in debugImages.keys():
                outputImWrite(f'{img_name}_{si}.jpg', debugImages[si])
            # current issue:
            #   far away ones can't be assigned -> do it in the 2nd round
            # tuneRhythm(noteGroupMap, stemIdxMap, noteGroupVerticallyMerged, restMap,restList,sfnClefMap,sfnClefList,beamMapImg,staffObjList)
            beamMapImg,noteGroupMap = extendNotegroupsToStaff(noteGroupMap, noteGroupVerticallyMerged,staffObjList,beamMapImg)
            barList, barRanges, numBarsPerLine = constructBar(noteGroupMap, stemIdxMap, noteGroupVerticallyMerged, restMap,restList,sfnClefMap,sfnClefList,beamMapImg,staffObjList)
            # maskImg, tsBoxes, tsBoxesFiltered, debugImages = createMask(barList, barRanges, staffObjList, image, beamMapImg)
            for si in debugImages.keys():
                outputImWrite(f'{img_name}_{si}.jpg', debugImages[si])
            # saveBarToCsv(img_name, barList, desiredLengtdh=1)
            # trackNo = 1
            # if img_wholename in is2:
            #     trackNo = 2
            # elif img_wholename in is5:
            #     trackNo = 5
            # elif img_wholename.startswith("quart"):
            #     trackNo = 4
            editBarTS(barList, config['tsChange'], imgIdx+1, numBarsPerLine)
            barBreakPoints = tuneBarList(barList, numBarsPerLine) # modifies barList
            trackNo = NUM_TRACK
            for tr in range(NUM_TRACK):
                wholeBarList[tr] += barList[tr]
            # getAccidentalsChanges(barList)
            score, scoreShifted, debugImages = exportXML(barList, trackNo, image, beamMapImg=beamMapImg, barsBreakPoints = barBreakPoints)
            if len(allBarBreakPoints) == 0:
                allBarBreakPoints+=barBreakPoints
            else:
                sumPrevBar = allBarBreakPoints[-1]
                allBarBreakPoints+=[bp+sumPrevBar for bp in barBreakPoints]
            with open(barList_path, "wb") as f:
                pickle.dump(barList, f)
            for si in debugImages.keys():
                outputImWrite(f'{img_name}_{si}.jpg', debugImages[si])
                imwrite(f'{si}.jpg', debugImages[si])
            score.write('musicxml', fp=f'{OUTPUT_BASE_FOLDER}/{img_name}.xml')
            print(f"MusicXML written to '{OUTPUT_BASE_FOLDER}/{img_name}.xml'")
            if len(np.unique(TRACK_SHIFT))>1:
                scoreShifted.write('musicxml', fp=f'{OUTPUT_BASE_FOLDER}/s_{img_name}.xml')
                print(f"MusicXML written to '{OUTPUT_BASE_FOLDER}/s_{img_name}.xml'")
            outputImWrite(f'DEBUG_{img_name}.jpg', DEBUGIMG)
            print(f'finishing parsing {img_name}')

        print('finishing all')
        scoreWhole, scoreWholeShifted, _= exportXML(wholeBarList, NUM_TRACK,barsBreakPoints=allBarBreakPoints)
        # scoreWhole, scoreWholeShifted, _= exportXML(wholeBarList, NUM_TRACK, beamMapList=allBeamMaps, beamMapRefList=beamRefList,lineNoList=lineNoList)
        scoreWhole.write('musicxml', fp=f'{OUTPUT_BASE_FOLDER}/all_{img_name}.xml')
        if len(np.unique(TRACK_SHIFT))>1:
            scoreWholeShifted.write('musicxml', fp=f'{OUTPUT_BASE_FOLDER}/whole_s_{img_name}.xml')
            print(f"MusicXML written to '{OUTPUT_BASE_FOLDER}/whole_s_{img_name}.xml'")

        print(f"MusicXML written to '{OUTPUT_BASE_FOLDER}/all_{img_name}.xml'")
        with open(f"{base_folder}{piece_name}/{piece_name}_barList.pkl","wb") as f:
            pickle.dump(wholeBarList, f)
# '''
        # TODO 
        # go through each staffline and get all elements on it:
        # assign length (for triplets etc.) based on horizontal grouping 
        #   if length adds to 1 or 2 -> keep it as it is 
        #   if approx 1 or 2: go through each's 

    
        # current data structure documentation: 
        # noteChunkList: list of None|NoteChunks 
        #   by noteChunkList[n].noteGroupIdxs you can get the Ids of them
        # noteGroupMap, stemIdxMap <-> noteGroupVerticallyMerged
        #   noteGroupMap value == 0: no note, 
        #   >0: is the noteGroupVerticallyMerged[idx]
        #   stemIdxMap value == -1: no noteBox
        #   >=0: is the index of noteGroupVerticallyMerged[idx].noteStemList[index]
        # restMap <-> restList
        #   RestMap value == 0: no rest
        #   >0: is the restList[idx]
        # sfnClefMap <-> sfnClefList
        #   sfnClefMap value == 0: no clef/sfn
        #   >0: is the sfnClefList[idx]
        # beamMapImg: bb,gg,rr 
        #   bb: the gradient beam image, 255 where there's beam, 
        #       254-2*stepSize*n: go how far down to beam 
        #       253-2*stepSize*n: go how far up to beam 
        #   gg: 0 if there's nothing there
        #       1 if it's place of a noteGroup object
        #       2 if it's place of an accidentals
        #       3 if it's place of a clef
        #       4 if it's place of a barline 
        #       5 if it's place of a rest
        #       6 if it's place of a noteGroup ornament? TODO not implemented yet
        #   rr: staff position map + staff number + how far to the bottom of staff
        #       x = 0~3 (total of 4): the position of the note (1~24, A is 12, C is 13) 
        #       x = 4: the staffnum of the closest one (starting with 1 instead of 0)
        #       x = -1: how far pixel to the top of the staff (if it's in the staff range and 1 barheight above, else 0) 
        #       x = -2: how far pixel to the bottom of the staff (if it's in the staff range and 1 barheight below, else 0) 
        #       within the x of staff: the staff number
        #           if it contains anything in the staff it will be that, else it will 


'''
NOTES
1. I've commented out the filters in part2.py
'''