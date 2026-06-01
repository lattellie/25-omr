import statistics
import cv2
import numpy as np

from typing import List,Tuple, Union, Set
from music21 import stream, note, metadata, chord, meter, clef, key
from scipy import stats

from utils import *

CLEF_OPTIONS = [[1],[1],[0,1],[-1,-2,1]]
TRACK_SHIFT = [0,0,0,0]

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
