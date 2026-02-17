from typing import List
import numpy as np
import pandas as pd
import csv
import os 
import json

from png2decode import png2decode
from utils import *



def constructBar(noteGroupMap:np.ndarray, 
                   stemIdxMap: np.ndarray,
                   noteGroupVerticallyMerged: List[NoteGroup|None],
                   restMap: np.ndarray,
                   restList:List[Rest|None],
                   sfnClefMap:np.ndarray,
                   sfnClefList:List[Union[Accidentals,Clef, None]],
                   beamMapImg:np.ndarray,
                   staffList:List[Staff]):
    numLines = len(staffList)
    bb,gg,rr = cv2.split(beamMapImg)
    barList:List[List[Bar]] = [[] for _ in range(numLines)]
    numBarsEachLine:List[int] = [0 for _ in range(numLines)]
    mapForMatching:List[np.ndarray|None] = [None,noteGroupMap,sfnClefMap, sfnClefMap, None, restMap]
    listForMatching:List[List] = [None, noteGroupVerticallyMerged,sfnClefList, sfnClefList, None, restList]
    notFinishedBar = None
    allRanges = []
    for kk in range(numLines):
        allBars = []
        sf0 = staffList[kk]
        barPlace = np.unique(np.where(gg[sf0.ys[0]:sf0.ys[-1], sf0.left:]==4)[1])
        isBar = np.sort(barPlace)
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
    for t in range(numLines): # trackNo
        printedLineNo = t # the 
        currBarList:List[Bar] = []
        sf0 = staffList[printedLineNo]
        ranges = allRanges[t]
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
        barList[t] = currBarList
        numBarsEachLine[t] = len(currBarList)
        # in order of vln1's 1st line, 2nd line ... | vln2's 1st line, 2nd line ...
    return barList,allRanges, numBarsEachLine

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
                )
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
    df = pd.read_csv(csvPath)
    pageMetadata = ScoreMetaData(df, jsonPath, True)
    instrumentList = ScoreMetaData.get_instruments()

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
    ) = png2decode(f"tchai_4_{number}", rf"orch_dataset\tchai_4\images\{number}\tchai_4_{number}.png")

    # get Barline locations -> bar center for each track: List[List[int]]
    # get the list of 
    print()



# todo:
# Timpani only use one row