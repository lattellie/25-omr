from abc import ABC, abstractmethod
from fractions import Fraction
import os
from typing import List,Tuple, Union, Set
from music21 import instrument
import math

import cv2
import numpy as np
import json

from dd_classes import RestNg

# --------------------------------------------------------------------------------------------------
# Setting for debug helper functions
# --------------------------------------------------------------------------------------------------
DEBUG_IMAGE = True
LOG_MESSAGE = True

# --------------------------------------------------------------------------------------------------
# Helper function for loggings
# --------------------------------------------------------------------------------------------------
def printt(str):
    if LOG_MESSAGE:
        print(str)

def imwrite(str,img,strictlyYes=False, startingFolder = ''):
    debugFolderPath = f'{startingFolder}debug'
    if not os.path.exists(debugFolderPath):
        os.mkdir(debugFolderPath)
    if DEBUG_IMAGE or strictlyYes:
        print(f'saving {str}')
        cv2.imwrite(f'{debugFolderPath}/{str}',img)


def outputImWrite(str,img, startingFolder = ''):
    outputImgPath = f'{startingFolder}output'
    if not os.path.exists(outputImgPath):
        os.mkdir(outputImgPath)
    cv2.imwrite(f'{outputImgPath}/{str}',img)

def writeDebugImagesFromDict(debugImages, writeToOutput = False, img_name = ''):
    for si in debugImages.keys():
        if writeToOutput and img_name != '':
            outputImWrite(f'{img_name}_{si}.jpg', debugImages[si])
        imwrite(f'{si}.jpg', debugImages[si])

# --------------------------------------------------------------------------------------------------
# helper function for Key Signatures
# --------------------------------------------------------------------------------------------------
class ToneHelper:
    def __init__(self, toneMapPath: str):
        with open(toneMapPath, 'r') as f:
            self.toneMap = json.load(f)
    # get the actual key signature based on what signature it looks like
    # for instance, in B flat clarinet, if it looks like 1 flat it's actually 3 flat, so -2
    def getKsShift(self, toneName: str, includeForKs = True) -> int:
        if toneName == "":
            return 0
        toneData = self.toneMap.get(toneName)
        if not includeForKs:
            return np.inf
        if toneData is None:
            print(f"can't find tone for {toneName}, set ksShift to 0")
            return 0
        else:
            return int(toneData[0])
    def getNoteShift(self, toneName: str) -> int:
        if toneName == "":
            return 0
        toneData = self.toneMap.get(toneName)
        if toneData is None:
            print(f"can't find tone for {toneName}, set noteShift to 0")
            return 0
        else:
            return int(toneData[1])
    def getKsAndNoteShift(self, toneName: str, includeForKs = True) -> Tuple[int, int]:
        if toneName == "":
            return (0,0)
        toneData = self.toneMap.get(toneName)
        noteShift = 0
        ksShift = np.inf
        if toneData is None:
            print(f"can't find tone for {toneName}, set all shifts to 0")
        else:
            noteShift = toneData[1]
            if includeForKs:
                ksShift = toneData[0]
        return ksShift, noteShift

# --------------------------------------------------------------------------------------------------
# Helper function for instrument
# --------------------------------------------------------------------------------------------------

def constructInstrumentMappingDict(json_file) -> dict:
    clef_map = {"treble": 1, "alto": 0, "bass": -1, "tenor": -2}
    with open(json_file, "r") as f:
        data = json.load(f)
    mapped = {}
    for instrument, clefs in data.items():
        mapped[instrument] = [clef_map[c] for c in clefs]
    return mapped

def isInstrumentIncluded(currIns: str):
    notIncluded = ["trumpet", "horn", "timpani"] # instruments that is noted in its own way
    return all(word not in currIns.lower() for word in notIncluded)

instrument_classes = {
    "violin": instrument.Violin,
    "viola": instrument.Viola,
    "cello": instrument.Violoncello,
    "double_bass": instrument.Contrabass,
    "bass": instrument.Contrabass,
    "piccolo": instrument.Piccolo,
    "flute": instrument.Flute,
    "oboe": instrument.Oboe,
    "english_horn": instrument.EnglishHorn,
    "clarinet": instrument.Clarinet,
    "bass_clarinet": instrument.BassClarinet,
    "bassoon": instrument.Bassoon,
    "contrabassoon": instrument.Contrabassoon,
    "french_horn": instrument.Horn,
    "trumpet": instrument.Trumpet,
    "trombone": instrument.Trombone,
    "bass_trombone": instrument.BassTrombone,
    "tuba": instrument.Tuba,
    "harp": instrument.Harp,
    "piano": instrument.Piano,
    "celesta": instrument.Celesta,
    "timpani": instrument.Timpani,
    "xylophone": instrument.Xylophone,
    "marimba": instrument.Marimba,
    "glockenspiel": instrument.Glockenspiel,
    "vibraphone": instrument.Vibraphone,
    "horn": instrument.Horn
}
def get_instrument_from_string(name: str):
    name_lower = name.lower()
    for key, instr_class in instrument_classes.items():
        if key in name_lower:
            return instr_class()
    return instrument.Piano()
# --------------------------------------------------------------------------------------------------
# Class definitions
# --------------------------------------------------------------------------------------------------
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
        self.indexNumber: int|None = None
    def setIndexNumber(self, index:int):
        self.indexNumber = index
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
        self.indexNumber: int|None = None
    def setIndexNumber(self, index:int):
        self.indexNumber = index
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
    if not ng1:
        return ng2
    if not ng2:
        return ng1

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
    def setIndexNumber(self, index:int):
        return
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
class KeySignature():
    def __init__(self, accs: List[Accidentals]):
        self.accs = accs
    def getSharpFlat(self)->float:
        return sum([acc.shift for acc in self.accs])/len(self.accs)
    # supposingly if there's 4 flats will be -4, 4 sharp will be +4
    def getSharpFlatInt(self) -> int:
        if sum([acc.shift for acc in self.accs]) > 0:
            return len(self.accs)
        else:
            return -len(self.accs)
    def setIndexNumber(self, index:int):
        return

    
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
    def __init__(self, TS:Tuple[int,int]=[4,4]):
        self.ts = TS
        self.elementList:List[Accidentals|Clef|Rest|NoteGroup] = []
        self.restNgList:List[RestNg] = []
    def assignEmptyBar(self):
        rng = RestNg(length=Fraction(self.ts[0], self.ts[1]),
                                groupId = -1,
                                isRest = True,
                                numNotes = 0,
                                beamLength = (-1,-1),
                                beamEnd=[-1,-1])
        emptyRest = Rest([-1,-1,-1,-1], -1)
        self.elementList.append(emptyRest)
        self.restNgList.append(rng)
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
