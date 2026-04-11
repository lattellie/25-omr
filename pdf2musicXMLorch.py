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
from music21 import stream, note, metadata, chord, meter, clef, key, instrument
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
        for i in range(1,4):
            df[f"combined{i}"] = (
                df[f"ins{i}"]
                + df[f"part{i}"].apply(lambda x: "" if pd.isna(x) else '__'+str(int(x)))
                + df[f"tone{i}"].apply(lambda x: "" if pd.isna(x) else ':'+x)
            ) # 'clarinet_1:B flat'
        # special case for timpani
        mask = df["ins1"] == "timpani"
        df.loc[mask, "combined1"] = "timpani"
        df.loc[mask, ["combined2", "combined3"]] = np.nan
        if initInstruments:
            df_filtered = df[['combined1', 'combined2','combined3']][df['system'] == 1]
            instrument_list = [str(x) for x in df_filtered.to_numpy().flatten() if pd.notna(x)]
            self.set_instruments(instrument_list)
            with open(self.jsonPath, "w", encoding="utf-8") as f:
                json.dump(instrument_list, f, ensure_ascii=False, indent=2)
        elif os.path.exists(self.jsonPath) and self.get_instruments() == []:
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
    
    # get the number of track that line belongs to
    def getTrackForLineIdx(self, lineIdx: int) -> int:
        for i, (start, end) in enumerate(self.trackRange):
            if start <= lineIdx <= end:
                return i
        return -1
    
    # get the line number of that line inside that track. 
    # let's say [3,6] are the 2nd group (inclusive), then 
    #   fn(3) -> (1,0), 2nd group(count starts at 0), index 0
    #   fn(6) -> (1,3), 2nd group, index 3
    def getTrackIdxForLineIdx(self, lineIdx: int) -> Tuple[int,int]:
        for i, (start, end) in enumerate(self.trackRange):
            if start <= lineIdx <= end:
                return (i, lineIdx - start)
        return (-1,-1)


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
        notfinishedBarStuff = np.where(gg[sf0.ys[0]-sf0.get_yOne():sf0.ys[-1]+sf0.get_yOne(), rng[1]:]>0)[1]
        nextIdx = 0
        if len(notfinishedBarStuff) > 0:
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

    from collections import defaultdict

    def build_grouped_lookup(instrument_list):
        grouped = defaultdict(list)
        for item in instrument_list:
            if ":" in item:
                base, suffix = item.split(":", 1)
                key_suffix = ":" + suffix
            else:
                base = item
                key_suffix = ""
            base_name = base.split("__")[0]
            key = base_name + key_suffix
            grouped[key].append(item)
        return dict(grouped)
    from collections import defaultdict

    def build_instrument_lookup_no_section(instrument_list):
        grouped = defaultdict(list)
        for item in instrument_list:
            base = item.split(":", 1)[0]
            base_name = base.split("__")[0]
            grouped[base_name].append(item)
        return dict(grouped)
    def getInstruments(instrumentLookup: dict, instrumentLookupNoSec: dict, ins: str):
        if instrumentLookup.get(ins):
            return instrumentLookup.get(ins)
        if instrumentLookupNoSec.get(ins):
            return instrumentLookupNoSec.get(ins)
        return [ins]
    for trackLineIdx, currTrackRange in enumerate(trackRange):
        startIdx = currTrackRange[0]
        endIdx = currTrackRange[1]+1
        instrumentList = ScoreMetaData.get_instruments()
        instrumentLookup = build_grouped_lookup(instrumentList)
        instrumentLookupNoSec = build_instrument_lookup_no_section(instrumentList)
        for i in range(startIdx, endIdx):
            instrumentInThisLineList = [getInstruments(instrumentLookup,instrumentLookupNoSec,ins) for ins in instrumentTrack[i]]
            instrumentInThisLine = [item for sublist in instrumentInThisLineList for item in sublist]
            print(f"instrument in this line: {instrumentInThisLine}")
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
    assert len(np.unique([len(retBarList[b]) for b in retBarList])) == 1
    barListPerInstrument = [retBarList[k] for k in list(retBarList.keys())]
    instrumentEachLine = list(retBarList.keys())
    return retBarList, barListPerInstrument, instrumentEachLine

# return (1) ksMat for each track (exactly as how it's presented visually), metric of [number of lines, bars]
#        (2) the keySignature of each track (concert pitch) as an array
def getKSMatAndKSList(pageMetadata: ScoreMetaData, barList: List[List[Bar]], toneHelper: ToneHelper):
    def checkAndAssignKS(
            prevIsAcc: bool, 
            new_elements: List[Accidentals|Clef|Rest|NoteGroup|KeySignature], 
            accumulatedAccidentals: List[KeySignature],
            trackGroupNo: int, 
            lineOfThatTrackNo: int, 
            barNo: int,
            ksBarMatForEachTrack: List[np.ndarray]):
        if prevIsAcc:
            newKS = KeySignature(accumulatedAccidentals)
            new_elements.append(newKS)
            ksBarMatForEachTrack[trackGroupNo][lineOfThatTrackNo][barNo] = newKS.getSharpFlatInt()
            accumulatedAccidentals = []
        return new_elements, accumulatedAccidentals, ksBarMatForEachTrack
    # return the list of concert pitch keySignature for each bar (inf: no signature)
    def modeKSForAllTrack(ksMat:np.ndarray):
        result = []
        prev = 0
        for col in ksMat.T:
            finite_vals = col[~np.isinf(col)]  # remove infs
            if finite_vals.size > 0:
                values, counts = np.unique(finite_vals, return_counts=True)
                prev = values[np.argmax(counts)]
            result.append(prev)
        return np.array(result)
    oneInstrumentEachLine: List[str] = [i[0] for i in pageMetadata.getInstrumentEachTrack()]
    tone: List[str] = [ins.split(':')[1] if len(ins.split(':')) > 1 else '' for ins in oneInstrumentEachLine]
    trackShiftEachLine = []
    for idx in range(len(oneInstrumentEachLine)):
        currTone: str = tone[idx]
        currIns: str = oneInstrumentEachLine[idx]
        if isInstrumentIncluded(currIns):
            trackShiftEachLine.append(-toneHelper.getKsShift(currTone))
        else:
            trackShiftEachLine.append(-np.inf)
    ksBarMatForEachTrack = []
    ksBarShiftForEachTrack = []
    trackRanges = pageMetadata.getTrackRange()
    for trackRange in trackRanges:
        numTrack = trackRange[1]-trackRange[0] + 1 # inclusive of the start and end
        numBars = len(barList[trackRange[0]])
        ksBarMat = np.ones((numTrack, numBars))*np.inf
        ksBarMatForEachTrack.append(ksBarMat)
        ksBarShift = np.array(trackShiftEachLine[trackRange[0]:trackRange[1]+1])[:, None] * np.ones(numBars)
        ksBarShiftForEachTrack.append(ksBarShift)

    for (lineNo, oneLine) in enumerate(barList):
        # the line #lineOfThatTrackNo of the track #trackGroupNo
        trackGroupNo, lineOfThatTrackNo = pageMetadata.getTrackIdxForLineIdx(lineNo)
        for (barNo, oneBar) in enumerate(oneLine):
            accumulatedAccidentals = []
            new_elements = []
            prevIsAcc = False
            for elem in oneBar.elementList:
                if isinstance(elem, Accidentals) and elem.isKeySignature:
                    accumulatedAccidentals.append(elem)
                    prevIsAcc = True
                else:
                    new_elements, accumulatedAccidentals, ksBarMatForEachTrack = checkAndAssignKS(prevIsAcc, new_elements, accumulatedAccidentals, trackGroupNo, lineOfThatTrackNo, barNo, ksBarMatForEachTrack)
                    new_elements.append(elem)
                    prevIsAcc = False
            new_elements, accumulatedAccidentals, ksBarMatForEachTrack = checkAndAssignKS(prevIsAcc, new_elements, accumulatedAccidentals, trackGroupNo, lineOfThatTrackNo, barNo, ksBarMatForEachTrack)
            # assign the basic to 0 if there's no changes
            if barNo == 0 and ksBarMatForEachTrack[trackGroupNo][lineOfThatTrackNo][barNo] == np.inf:
                ksBarMatForEachTrack[trackGroupNo][lineOfThatTrackNo][barNo] = 0
            oneBar.elementList = new_elements
    # now ksBarMatForEachTrack is exactly what we see on the score
    ksEachTrack = []
    ksMatEachTrackSameAsVisual = []
    for idx in range(len(ksBarMatForEachTrack)):
        ksBarMatShifted = ksBarMatForEachTrack[idx]-ksBarShiftForEachTrack[idx]
        modeKSArray = modeKSForAllTrack(ksBarMatShifted)
        ksEachTrack.append(modeKSArray)
        ksMatEachTrackSameAsVisual.append(np.tile(modeKSArray, (ksBarMatForEachTrack[idx].shape[0],1)) + ksBarShiftForEachTrack[idx])
    return ksMatEachTrackSameAsVisual, ksEachTrack, barList

def exportXML(barList:List[List[Bar]],  
              instrumentEachLine: List[str],
              toneHelper: ToneHelper,
              instrument_dict: dict,
              ksListAllTrack: np.ndarray, # array([-4., inf, inf, inf, inf, inf, 2, inf, inf])
              image:np.ndarray|None = None,
              imgName: str = "Title"):
    
    def getTrackKsAndNoteShift(toneHelper: ToneHelper, instrumentList: List[str]) -> Tuple[List[int|None], List[int]]:
        tone = [ins.split(':')[1] if len(ins.split(':')) > 1 else '' for ins in instrumentList]
        trackKsShift = []
        trackNoteShift = []
        for idx, tn in enumerate(tone):
            ksShift, noteShift = toneHelper.getKsAndNoteShift(tn)
            if isInstrumentIncluded(instrumentList[idx]):
                trackKsShift.append(-ksShift)
            else:
                trackKsShift.append(None)
            trackNoteShift.append(noteShift)
        return trackKsShift, trackNoteShift
    numTrack = len(instrumentEachLine)
    # Stem's label -> rhythm meaning
    # class_colors = [(255,0,0),(0,0,255),(255,255,0),(0,120,255),(40,255,40), (245, 220, 255),(230,130,175),(165,170, 70)]
    if image is None:
        debugXMLImg = None
    else:
        debugXMLImg = image.copy()
    def initializeScore(instrumentList: List[str], title):
        score = stream.Score()
        score.metadata = metadata.Metadata()
        score.metadata.title = title
        part = []
        for instr_name in instrumentList:
            p = stream.Part()
            # Assign instrument
            instr_obj = get_instrument_from_string(instr_name)
            p.insert(0, instr_obj)
            p.partName = instr_name
            p.partAbbreviation = instr_name
            part.append(p)
        return score, part
    score, part = initializeScore(instrumentEachLine, imgName)
    score2, part2 = initializeScore(instrumentEachLine, f"{imgName} Shifted")
    keyName = ['C','D','E','F','G','A','B']
    keyCharFlat = ['C','D-','D','E-','E','F','G-','G','A-','A','B-','B']
    keyCharSharp = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    currentClef = [None]*numTrack
    sharps = [3,0,4,1,5,2,6] # F C G D A E B
    flats = [6,2,5,1,4,0,3] # B E A D G C F
    ksSharp = [set() for _ in range(numTrack)]
    ksFlat = [set() for _ in range(numTrack)]
    trksOriginal = [[] for _ in range (numTrack)] # for tracking all original notes we see
    trksShifted = [[] for _ in range (numTrack)] # for tracking all shifted notes

    # trackKsShift will be None for Timpani, Horn and Trumpet
    # # trackNoteShift works for everything except for Timpani
    trackKsShift, trackNoteShift = getTrackKsAndNoteShift(toneHelper, instrumentEachLine)

    cleanedInstrumentList = [s.split('__')[0].split(':')[0] for s in instrumentEachLine]
    clefOptionEachLine = [instrument_dict.get(instr, [1,0,-1,-2]) for instr in cleanedInstrumentList]

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
    numBars = len(barList[0])
    clefMat = np.ones((numTrack, numBars))*np.inf 
    currentClef = [None]*numTrack
    for currBarNumber in range(numBars):
        barNumber = currBarNumber+1
        for lineNo in range(len(barList)):
            if len(clefOptionEachLine[lineNo]) == 1:
                clefMat[lineNo, currBarNumber] = clefOptionEachLine[lineNo][0]
            else:
                currBar = barList[lineNo][currBarNumber]
                for elem in currBar.elementList:
                    if type(elem) == Clef:
                        newClef = parseClefOne(elem, lineNo)
                        if newClef in clefOptionEachLine[lineNo]:
                            clefMat[lineNo, currBarNumber] = newClef
                if clefMat[lineNo, currBarNumber] == np.inf:
                    if currBarNumber == 0:
                        clefMat[lineNo, currBarNumber] = clefOptionEachLine[lineNo][0]
                    else:
                        clefMat[lineNo, currBarNumber] = clefMat[lineNo, currBarNumber-1]
    global DEBUGIMG
    ksListAllTrackNoInf = ksListAllTrack.copy()
    for i in range(1, numBars):
        if np.isinf(ksListAllTrackNoInf[i]):
            ksListAllTrackNoInf[i] = ksListAllTrackNoInf[i - 1]
    ksBarMatNeutral = np.tile(ksListAllTrackNoInf, (numTrack,1))   
    ksBarMatNeutral = ksBarMatNeutral.astype(int)    
    ksBarMatNew = ksBarMatNeutral + np.tile(np.array(TRACK_SHIFT),(numBars,1)).T
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

def numberToString(num: int, strLen: int = 3):
    stringNum = str(num)
    strAppend = strLen-len(stringNum)
    return '0'*strAppend + stringNum

def getTrackKsAndNoteShift(toneHelper: ToneHelper, instrumentList: List[str]) -> Tuple[List[int|None], List[int]]:
    tone = [ins.split(':')[1] if len(ins.split(':')) > 1 else '' for ins in instrumentList]
    trackKsShift = []
    trackNoteShift = []
    for idx, tn in enumerate(tone):
        ksShift, noteShift = toneHelper.getKsAndNoteShift(tn)
        if isInstrumentIncluded(instrumentList[idx]):
            trackKsShift.append(-ksShift)
        else:
            trackKsShift.append(None)
        trackNoteShift.append(noteShift)
    return trackKsShift, trackNoteShift

# time signatures
def getMeasuresEachStaff(image: np.ndarray, beamMapImg: np.ndarray, staffList: List[Staff]):
    img = image.copy()
    # 第一維：哪一行 staff (長度等於 len(staffList))
    allStavesMeasures = [[] for _ in range(len(staffList))]
    
    # 取得類別 Map 並只關注小節線類別 (4)
    _, itemMap, _ = cv2.split(beamMapImg)
    staffCenters = [sf.ys[2] for sf in staffList]
    staffImgBinary = (itemMap == 4)[staffCenters, :]
    
    # 利用 TrackRange 偵測，因為同一個 Track 的小節線通常是垂直對齊的
    for trkRange in pageMetadata.getTrackRange():
        staves_in_track = range(trkRange[0], trkRange[1] + 1)
        
        # 投影判定小節線位置
        barSum = np.sum(staffImgBinary[trkRange[0]:trkRange[1]+1, :], axis=0)
        numStaves = len(staves_in_track)
        barPos = np.where(barSum > numStaves * 0.7)[0]
        
        if len(barPos) > 0:
            # --- 1. 篩選小節線：連續像素內 (<=5) 僅保留最右邊的一個值 ---
            barLinesX = []
            if len(barPos) > 0:
                temp_group_max = barPos[0]
                for i in range(1, len(barPos)):
                    if barPos[i] - barPos[i-1] <= BAR_MAX_GAP:
                        temp_group_max = barPos[i] # 持續更新，保留最右邊
                    else:
                        barLinesX.append(temp_group_max)
                        temp_group_max = barPos[i]
                barLinesX.append(temp_group_max)
            
            # --- 2. 根據小節線生成「小節 (Measures)」 ---
            # 如果有 N 條線 (barLinesX)，產生 N-1 個小節
            if len(barLinesX) >= 2:
                for m_idx in range(len(barLinesX) - 1):
                    m_left = barLinesX[m_idx]
                    m_right = barLinesX[m_idx+1]
                    
                    # 分配給該 Track 裡的每一行 Staff
                    for s_idx in staves_in_track:
                        target_staff = staffList[s_idx]
                        top = target_staff.ys[0]
                        bottom = target_staff.ys[-1]
                        
                        # 建立小節 Bounding Box [left, right, top, bottom]
                        measure_bbox = [int(m_left), int(m_right), int(top), int(bottom)]
                        allStavesMeasures[s_idx].append(measure_bbox)
                        
                        # 視覺化：畫出小節範圍 (用藍色框表示小節)
                        cv2.rectangle(img, (m_left, top), (m_right, bottom), (255, 0, 0), 1)

    cv2.imwrite("measures_detected.jpg", img)
    return allStavesMeasures

def detect_time_signatures(model_path: str, img_path: str, imgsz: int = 1792, conf: float = 0.5) -> List[TimeSignature]:
        from ultralytics import YOLO
        time_signatures = []

        model = YOLO(str(model_path))
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf,
            save=False,
            verbose=False
        )

        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            clss   = r.boxes.cls.cpu().tolist()
            confs  = r.boxes.conf.cpu().tolist()
            xyxys  = r.boxes.xyxy.cpu().tolist()

            for i in range(len(clss)):
                cls_id = int(clss[i])
                conf_val = float(confs[i])
                x1, y1, x2, y2 = xyxys[i]
                bbox = (int(x1), int(y1), int(x2), int(y2))
                ts_obj = TimeSignature(yolo_class=cls_id, bbox=bbox, confidence=conf_val)
                time_signatures.append(ts_obj)

        return time_signatures

def map_time_signatures_to_score(time_signature_list, staffList, allStavesMeasures, allItemInScore):
    """
    將 TimeSignature 物件映射到對應的 Staff 和 Measure。
    """
    for ts in time_signature_list:
        if not ts.isValid():
            continue

        tx1, ty1, tx2, ty2 = ts.getBbox()
        ts_cx = (tx1 + tx2) / 2  # 中心 X
        ts_cy = (ty1 + ty2) / 2  # 中心 Y
        ts_h = ty2 - ty1         # bbox 高度

        # --- 規則 1: 縱向尋找最靠近的 staff ---
        best_staff_idx = -1
        min_dist = float('inf')

        for s_idx, staff in enumerate(staffList):
            # 計算該 staff 的中心 Y (五條線的平均值)
            staff_cy = sum(staff.ys) / len(staff.ys)
            dist = abs(ts_cy - staff_cy)
            
            if dist < min_dist:
                min_dist = dist
                best_staff_idx = s_idx

        # 檢查垂直相差距離是否大過本身 bbox 高度
        if best_staff_idx == -1 or min_dist > ts_h:
            ts.setValid(False)
            continue

        ts.setstaffline(best_staff_idx)

        # --- 規則 2: 水平尋找小節 (Measure) ---
        measures = allStavesMeasures[best_staff_idx]
        found_bar_idx = -2 # 預設未找到

        # 檢查是否落在某個小節內
        for m_idx, m_bbox in enumerate(measures):
            mx1, mx2, my1, my2 = m_bbox
            if mx1 <= ts_cx <= mx2:
                found_bar_idx = m_idx
                break
        
        # 如果沒在小節內，檢查是否在最後一個小節的右邊
        if found_bar_idx == -2:
            last_mx2 = measures[-1][1] # 最後一個小節的 x2
            if ts_cx > last_mx2:
                found_bar_idx = -1
            else:
                # 既不在小節內，也不在右邊（可能在第一小節左邊邊緣外）
                ts.setValid(False)
                continue

        ts.setBarLoc(found_bar_idx)

        # --- 規則 3: 檢查小節內的 Rest 和 NoteGroup ---
        # 如果 barLoc 為 -1 (在最右邊之外)，通常不具備音樂意義上的音符檢查需求，可視情況跳過或判定無效
        if found_bar_idx == -1:
            # 依據一般邏輯，拍號不應出現在所有音符之後的最右側，建議設為無效
            ts.setValid(False)
            continue

        # 獲取該 staff 的所有物件
        staff_items = allItemInScore[best_staff_idx]
        mx1, mx2, _, _ = measures[found_bar_idx]

        is_invalid_by_context = False
        for item in staff_items:
            # 只考慮 Rest 和 NoteGroup (透過類別名稱判斷)
            class_name = item.__class__.__name__
            if class_name in ["Rest", "NoteGroup"]:
                ix1, iy1, ix2, iy2 = item.getBbox()
                item_cx = (ix1 + ix2) / 2
                
                # 判斷該物件是否屬於當前小節 (中心在小節範圍內)
                if mx1 <= item_cx <= mx2:
                    # 如果物件的中心在拍號中心的左邊 -> 拍號位置錯誤
                    if item_cx < ts_cx:
                        is_invalid_by_context = True
                        break
        
        if is_invalid_by_context:
            ts.setValid(False)
            continue
    

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
    sheetName = 'Bee_5_challenge'
    lenString = 3
    for number in range(1,5):
        csvPath = rf"orch_dataset\{sheetName}\csv\{sheetName}_{numberToString(number,lenString)}.csv"
        jsonPath = rf"orch_dataset\{sheetName}\csv\{sheetName}.json"
        df = pd.read_csv(csvPath)
        pageMetadata = ScoreMetaData(df, jsonPath, number == 1)
        instrumentList = ScoreMetaData.get_instruments()
        imgName = rf"{sheetName}_{number}"
        imgPath = rf"orch_dataset\{sheetName}\imgs\{sheetName}_{number}\{sheetName}_{number}.png"
        pklPath = rf"orch_dataset\{sheetName}\imgs\{sheetName}_{number}\{sheetName}_{number}.pkl"
        scorePath = rf"orch_dataset\{sheetName}\xmls\{sheetName}_{number}.musicxml"
        scoreShiftedPath = rf"orch_dataset\{sheetName}\xmls\{sheetName}_{number}_shifted.musicxml"
        if not os.path.isdir(rf"orch_dataset\{sheetName}\xmls"):
            os.mkdir(rf"orch_dataset\{sheetName}\xmls")
        
        if not os.path.exists(pklPath):
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
            ) = png2decode(imgName, imgPath)
            with open(pklPath, "wb") as f:
                pickle.dump(
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
                    ),
                    f
                )
            print(f"finishing processing sheet {sheetName}_{number}")
        # to save time running previous step (Debug only)

        with open(pklPath, "rb") as f:
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

        allStavesMeasures = getMeasuresEachStaff(image, beamMapImg, staffList)

        # load in image
        img = cv2.imread(imgPath)
        img_height, img_width, channels = img.shape

        # yolo detect time signature
        yolo_model_path = "best_time_signature.pt"
        time_signature_list = detect_time_signatures(yolo_model_path, imgPath)

        # filter for valid time signatures only
        map_time_signatures_to_score(time_signature_list, staffList, allStavesMeasures, allItemInScore)
        valid_ts_only = [ts for ts in time_signature_list if ts.isValid()]
        time_sig_thres = 0.5 
        final_time_sig_list = [
            [(ts.staffline, ts.barLoc), ts.getString()] 
            for ts in valid_ts_only 
            if ts.getConfidence() >= time_sig_thres
        ]
        final_time_sig_list.sort(key=lambda x: (x[0][0], x[0][1]))
        print("Detected Time Signatures (after filtering):", final_time_sig_list)

        # TODO: add in the bar time signature, [[(9.8),(9,8)...], [(9.8),(9,8),(4,4)...]] etc.
        barChangeList = dict()
        barChangeList['0,0'] = [2,4] # (track, num),(TStop, TSbottom)
        # get the list of barList (untuned)
        barList,allRanges, numBarsEachLine = constructBar(noteGroupMap, noteGroupVerticallyMerged, restMap ,restList, sfnClefMap, sfnClefList, beamMapImg, staffList, pageMetadata, barEachTrack, barChangeList)

        numBarsEachTrack = [numBarsEachLine[p[0]] for p in pageMetadata.getTrackRange()]

        # tune bar list based on timeSignature
        barBreakPoints = tuneBarList(barList, numBarsEachTrack)
        
        toneHelper = ToneHelper("keySignatureMapping.json")
        
        # returned BarList is exactly what we see in the score (two instrument in one line etc.)
        ksMatEachTrackDirectMap, ksEachTrack, barList = getKSMatAndKSList(pageMetadata, barList, toneHelper)
        
        # add instruments tracks for those not appearing
        # all instrument seperated (and added if not present)
        barDictPerInstrument, barListPerInstrument, instrumentEachLine = assignBarlistInstrument(barList, pageMetadata)

        ksListAllTrack = np.concatenate(ksEachTrack)
        assert len(ksListAllTrack) == len(barListPerInstrument[0])
        
        instrument_dict = constructInstrumentMappingDict("instrumentMapping.json")
        score, scoreShifted, debugImages = exportXML(barListPerInstrument, instrumentEachLine, toneHelper, instrument_dict, ksListAllTrack, image, imgName)
        score.write('musicxml', scorePath)
        print(f"score write to {scorePath}")
        scoreShifted.write('musicxml',scoreShiftedPath)
        print(f"shifted score write to {scoreShiftedPath}")
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
