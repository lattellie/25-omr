from typing import List
import numpy as np
import pandas as pd
import csv
import os 
import json
import pickle
import itertools
import statistics

from collections import Counter
from music21 import stream, note, metadata, chord, meter, clef, key
from scipy import stats

from png2decode import png2decode
from utils import *
from tune_bar import tuneBar


BAR_MAX_GAP = 5 # if two "lines" are within this pixel then they are considered the same barline
DEBUGIMG = None

class ScoreMetaData:
    # static 
    instruments: List[str] = []

    def __init__(self, pd_frame: pd.DataFrame, json_path: str, initInstruments = False):
        self.jsonPath: str = json_path
        self.pdFrame: pd.DataFrame = pd_frame
        self.trackRange: List[Tuple[int,int]] = []
        self.groupingRange: List[List[Tuple[int,int]]] = []
        if initInstruments:
            for i in range(1,4):
                df[f"combined{i}"] = (
                    df[f"ins{i}"]
                    + df[f"part{i}"].apply(lambda x: "" if pd.isna(x) else '_'+str(int(x)))
                    + df[f"tone{i}"].apply(lambda x: "" if pd.isna(x) else ':'+x)
                ) # 'clarinet_1:B flat'
            df_filtered = df[['combined1', 'combined2','combined3']][df['system'] == 1]
            instrument_list = [str(x) for x in df_filtered.to_numpy().flatten() if pd.notna(x)]
            self.set_instruments(instrument_list)
            with open(self.jsonPath, "w", encoding="utf-8") as f:
                json.dump(instrument_list, f, ensure_ascii=False, indent=2)

        elif os.path.exists(self.jsonPath) and self.get_instruments == []:
            with open(self.jsonPath, "r", encoding="utf-8") as f:
                instrument_list = json.load(f)
                self.set_instruments(instrument_list)
        

    def getTrackRange(self) -> List[Tuple[int,int]]:
        if len(self.trackRange)!= 0:
            return self.trackRange
        col = self.pdFrame["system"]
        groups = col.ne(col.shift()).cumsum()
        ranges = (
            self.pdFrame
            .groupby(groups)
            .apply(lambda g: (g.index[0], g.index[-1]))
            .tolist()
        )
        self.trackRange = ranges
        return ranges

    def getGroupingRange(self)->List[List[Tuple[int,int]]]:
        if len(self.groupingRange)!= 0:
            return self.groupingRange
        range1 = self.getTrackRange()

        col2 = self.pdFrame["staffgroup"]
        groups2 = col2.ne(col2.shift()).cumsum()
        ranges2 = (
            self.pdFrame
            .groupby(groups2)
            .apply(lambda g: (g.index[0], g.index[-1]))
            .tolist()
        )
        rangeList = []
        currentIdx = 0
        for r in range1:
            rangeInLine = []
            while currentIdx < len(ranges2) and ranges2[currentIdx][1] <= r[1]:
                rangeInLine.append(ranges2[currentIdx])
                currentIdx += 1
            rangeList.append(rangeInLine)
        self.groupingRange = rangeList
        return rangeList
    def getInstrumentEachTrack(self) -> List[List[str]]:
        return self.pdFrame[['combined1', 'combined2', 'combined3']].apply(
            lambda row: row.dropna().tolist(),
            axis=1
        ).tolist()
    def getInstrumentIndexEachTrack(self) -> List[List[int]]:
        strList = self.getInstrumentEachTrack()
        insList = self.get_instruments()
        return [[insList.index(r) for r in row] for row in strList]
    def getTrackForLineIdx(self, lineIdx: int) -> int:
        for i, (start, end) in enumerate(self.trackRange):
            if start <= lineIdx <= end:
                return i
        return -1

    # static
    @classmethod
    def set_instruments(cls, instrumentList: List[str]):
        cls.instruments = instrumentList


    @classmethod
    def add_instrument(cls, instrument: str):
        cls.instruments.append(instrument)

    @classmethod
    def get_instruments(cls) -> List[str]:
        return cls.instruments.copy()
    
def parseCsvData(csv_path: str):
    with open(csv_path) as f:
        csvData = list(csv.reader(f))
    return csvData

def getBarsEachTrack(image:np.ndarray, beamMapImg:np.ndarray, staffList:List[Staff]):
    img = image.copy()
    staffCenters = [sf.ys[2] for sf in staffList]
    _,itemMap,_ = cv2.split(beamMapImg)
    staffImgBinary = (itemMap == 4)[staffCenters,:]
    barEachTrack = []
    for trkRange in pageMetadata.getTrackRange():
        staffStart = staffList[trkRange[0]].ys[0]
        staffEnd = staffList[trkRange[1]].ys[-1]
        barSum = np.sum(staffImgBinary[trkRange[0]:trkRange[1],:],axis=0)
        barPos = np.where(barSum>(trkRange[1]-trkRange[0])*0.7)
        img[staffStart:staffEnd, barPos] = (0,0,255)
        barEachTrack += barPos
    imwrite("barForTracks.jpg", img)
    return barEachTrack


def getAllObjectInEachLine(
        noteGroupMap: np.ndarray,
        noteGroupVerticallyMerged: List[NoteGroup | None],
        restMap: np.ndarray,
        restList: List[Rest | None],
        sfnClefMap: np.ndarray,
        sfnClefList: List[Union[Accidentals, Clef, None]],
        beamMapImg: np.ndarray,
        staffList: List[Staff]
    ):
    _, objMap, _ = cv2.split(beamMapImg)
    mapForMatching:List[np.ndarray|None] = [None,noteGroupMap,sfnClefMap, sfnClefMap, None, restMap]
    listForMatching:List[List] = [None, noteGroupVerticallyMerged,sfnClefList, sfnClefList, None, restList]
    allItems = []
    for sf in staffList:
        allItemInLine = []
        currX = sf.left
        while currX < sf.right:
            lineLst = np.unique(objMap[sf.ys[0]-sf.get_yOne():sf.ys[-1]+sf.get_yOne(), currX]).tolist()
            if 0 in lineLst:
                lineLst.remove(0)
            if len(lineLst)>1:
                for i in [1,5,3,2]:
                    if i in lineLst:
                        lineLst = [i]
                        break
            if len(lineLst) == 0 or 4 in lineLst:
                currX += 1
            else:
                typeId = lineLst[0]
                currMap = mapForMatching[typeId]
                currLst = listForMatching[typeId]
                inMapId = np.unique(currMap[sf.ys[0]-sf.get_yOne():sf.ys[-1]+sf.get_yOne(), currX]).tolist()
                if 0 in inMapId:
                    inMapId.remove(0)
                if (len(inMapId) == 1):
                    allItemInLine.append(currLst[inMapId[0]])
                    currX = currLst[inMapId[0]].boundingBox[2]+1
                else:
                    currX += 1
        allItems.append(allItemInLine)
    return allItems

def constructBar(noteGroupMap:np.ndarray, 
                   noteGroupVerticallyMerged: List[NoteGroup|None],
                   restMap: np.ndarray,
                   restList:List[Rest|None],
                   sfnClefMap:np.ndarray,
                   sfnClefList:List[Union[Accidentals,Clef, None]],
                   beamMapImg:np.ndarray,
                   staffList:List[Staff],
                   pageMetadata:ScoreMetaData,
                   barEachTrack:List[List[int]],
                   barChangeList:dict[str, tuple[int, int]] #(track, num),(TStop, TSbottom)
                   ): 
    numLines = len(staffList)
    bb,gg,rr = cv2.split(beamMapImg)
    barList:List[List[Bar]] = [[] for _ in range(numLines)]
    numBarsEachLine:List[int] = [0 for _ in range(numLines)]
    mapForMatching:List[np.ndarray|None] = [None,noteGroupMap,sfnClefMap, sfnClefMap, None, restMap]
    listForMatching:List[List] = [None, noteGroupVerticallyMerged,sfnClefList, sfnClefList, None, restList]
    notFinishedBar = None
    allRanges = []
    for barThisTrack in barEachTrack:
        prevIdx = 0
        rangeThisTrack = []
        if len(barThisTrack) > 1:
            while prevIdx < len(barThisTrack)-1:
                if barThisTrack[prevIdx+1] - barThisTrack[prevIdx] > BAR_MAX_GAP:
                    rangeThisTrack.append((barThisTrack[prevIdx], barThisTrack[prevIdx+1]))
                prevIdx += 1
        allRanges.append(rangeThisTrack)

    currTS = barChangeList.get('0,0')
    for t in range(numLines): # trackNo
        printedLineNo = t # the 
        currBarList:List[Bar] = []
        sf0 = staffList[printedLineNo]
        trackForThatLine = pageMetadata.getTrackForLineIdx(t)
        ranges = allRanges[trackForThatLine]
        isBar = barEachTrack[trackForThatLine]
       
        for ridx, rng in enumerate(ranges):
            if barChangeList.get(f'{trackForThatLine},{ridx}'):
                currTS = barChangeList.get(f'{trackForThatLine},{ridx}')
                if currTS != [9,8]:
                    print(f'ridx: ${ridx}, rng: ${rng}')
            if notFinishedBar is not None:
                bar = notFinishedBar
                notFinishedBar = None
            else:
                bar = Bar(currTS)
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
            bar = Bar(currTS)
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
        barList[t] = currBarList
        numBarsEachLine[t] = len(currBarList)
        # in order of vln1's 1st line, 2nd line ... | vln2's 1st line, 2nd line ...
    return barList,allRanges, numBarsEachLine

# originally we pass in a list of each track [vln1's bars: [bar1, bar2 ...], vln2's bars: [bar1, bar2 ...]]
# now we have each line differently
# return the breakPoint (aka which bar is the starting of next line)
def tuneBarList(barList:List[List[Bar]], numBarsPerTrackLine:List[int], ):
    trackRange = pageMetadata.getTrackRange()

    for lineRangeForTrack in trackRange:
        startIdx = lineRangeForTrack[0]
        endIdx = lineRangeForTrack[1]+1
        emptyBarsIndexs = np.where(np.sum(np.array([[len(b.elementList) for b in oneTrack] for oneTrack in barList[startIdx:endIdx]]),axis=0)==0)[0].tolist()
        emptyBarsIndexs.reverse()
        for emptyBarsIndex in emptyBarsIndexs:
            for i in range(startIdx, endIdx):
                del barList[i][emptyBarsIndex]
            qq = 0
            while emptyBarsIndex+1>sum(numBarsPerTrackLine[0:qq]):
                qq+=1
            numBarsPerTrackLine[qq-1]-=1
    
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
    barBreakPoints = [0] + list(itertools.accumulate(numBarsPerTrackLine))
    return barBreakPoints

def assignBarlistInstrument(barList:List[List[Bar]], pageMetadata:ScoreMetaData):
    instrumentTrack = pageMetadata.getInstrumentEachTrack()
    trackRange = pageMetadata.getTrackRange()
    retBarList = dict()
    def getEmptyBarDefaultList(barTrack: List[Bar]):
        return [Bar(b.ts) for b in barTrack]

    for trackLineIdx, currTrackRange in enumerate(trackRange):
        instrumentList = ScoreMetaData.get_instruments()
        startIdx = currTrackRange[0]
        endIdx = currTrackRange[1]+1
        for i in range(startIdx, endIdx):
            instrumentInThisLine = instrumentTrack[i]
            for ins in instrumentInThisLine:
                if trackLineIdx == 0:
                    retBarList[ins] = list(barList[i])
                else:
                    retBarList[ins] += barList[i]
                if ins in instrumentList:
                    instrumentList.remove(ins)
                else:
                    print(f"instrument {ins} not in list")
        for resIns in instrumentList:
            if trackLineIdx == 0:
                retBarList[resIns] = getEmptyBarDefaultList(barList[startIdx])
            else:
                retBarList[resIns] += getEmptyBarDefaultList(barList[startIdx])
    return retBarList
def exportXML(barList:List[List[Bar]], numTrack:int, 
              TRACK_SHIFT: List[int],
              CLEF_OPTIONS: List[List[int]],
              image:np.ndarray|None = None, 
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

if __name__ == '__main__':
    noteGroupMap: np.ndarray
    stemIdxMap: np.ndarray
    noteGroupVerticallyMerged: List[NoteGroup | None]
    restMap: np.ndarray
    restList: List[Rest | None]
    sfnClefMap: np.ndarray
    sfnClefList: List[Union[Accidentals, Clef, None]]
    beamMapImg: np.ndarray
    staffList: List[Staff]
    number = '020'

    csvPath = rf"orch_dataset\tchai_4\csv\tchai_4_{number}.csv"
    jsonPath = csvPath.replace('.csv','.json')
    imgPath = rf"orch_dataset\tchai_4\images\{number}\tchai_4_{number}.png"
    df = pd.read_csv(csvPath)
    pageMetadata = ScoreMetaData(df, jsonPath, True)
    instrumentList = ScoreMetaData.get_instruments()

    # # actual use
    # (
    #     noteGroupMap,
    #     stemIdxMap,
    #     noteGroupVerticallyMerged,
    #     restMap,
    #     restList,
    #     sfnClefMap,
    #     sfnClefList,
    #     beamMapImg,
    #     staffList,
    #     dataDict
    # ) = png2decode(f"tchai_4_{number}", rf"orch_dataset\tchai_4\images\{number}\tchai_4_{number}.png")
    
    # to save time running previous step (Debug only)
    with open("processedData.pkl", "rb") as f:
        (
            noteGroupMap,
            stemIdxMap,
            noteGroupVerticallyMerged,
            restMap,
            restList,
            sfnClefMap,
            sfnClefList,
            beamMapImg,
            staffList,
            dataDict
        ) = pickle.load(f)

    # !!! decoded stuff from score (by bar)
    # !!! note object by staff line (object not just decoded)
    allItemInScore = getAllObjectInEachLine(noteGroupMap, noteGroupVerticallyMerged, restMap, restList, sfnClefMap, sfnClefList, beamMapImg, staffList)

        
    image = dataDict['image']
    # get Barline locations -> bar center for each track: List[List[int]]
    barEachTrack = getBarsEachTrack(image, beamMapImg, staffList)
    # TODO: add in the bar time signature, [[(9.8),(9,8)...], [(9.8),(9,8),(4,4)...]] etc.
    barChangeList = dict()
    barChangeList['0,0'] = [9,8] # (track, num),(TStop, TSbottom)
    # get the list of barList (untuned)
    barList,allRanges, numBarsEachLine = constructBar(noteGroupMap, noteGroupVerticallyMerged, restMap ,restList, sfnClefMap, sfnClefList, beamMapImg, staffList, pageMetadata, barEachTrack, barChangeList)

    numBarsEachTrack = [numBarsEachLine[p[0]] for p in pageMetadata.getTrackRange()]

    # tune bar list based on timeSignature
    barBreakPoints = tuneBarList(barList, numBarsEachTrack)

    # add instruments tracks for those not appearing
    barDictPerInstrument = assignBarlistInstrument(barList, pageMetadata)
    barListPerInstrument = [barDictPerInstrument[k] for k in list(barDictPerInstrument.keys())]
    INSTRUMENT_TRK = list(barDictPerInstrument.keys())
    # TODO: enter clef options for each instrument (ex: flute - 1(treble), viola - 0(alto), cello-[1,-1,-2](treble, bass, tenor))
    CLEF_OPTIONS = [[1,0,-1,-2] for _ in barListPerInstrument]
    tone = [ins.split(':')[1] if len(ins.split(':')) > 1 else '' for ins in INSTRUMENT_TRK]
    TRACK_SHIFT = []
    for tn in tone:
        if 'B flat' in tn:
            TRACK_SHIFT.append(2)
        elif 'F' in tn:
            TRACK_SHIFT.append(4)
        else:
            TRACK_SHIFT.append(0)
    score, scoreShifted, debugImages = exportXML(barListPerInstrument, len(INSTRUMENT_TRK), TRACK_SHIFT, CLEF_OPTIONS, image, beamMapImg=beamMapImg, barsBreakPoints = barBreakPoints)

    print()


            
                





# todo:
# Timpani only use one row


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
