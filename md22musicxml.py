from music21 import stream, note, metadata, chord, meter, clef, key, instrument
import re
from typing import Tuple, List
from fractions import Fraction
from music21.musicxml.m21ToXml import ScoreExporter
from xml.etree.ElementTree import tostring
def processLine(lineStr: str):
    match lineStr[0]:
        case " ":
            pass
def printt(st):
    print(st)
def printtt(st):
    print(f"      --- {st}")

def updatePartName(part, name):
    part.partName = name

class ParserState:
    def __init__(self):
        self.isBeginningOfTrack: bool = True
        self.isDirty: bool = True
        self.currentInstrument: str = ''
        self.currClef: clef = clef.TrebleClef()
        self.currPart: stream.Part = stream.Part()
        # quarterNum = 3: length 3 is a quarter note, length 1 is a triplet
        self.quarterNum: int = 0
        # 3: ###, -4: bbbb
        self.keySigature: int = 0
        self.timeSignature: Tuple[int,int] = [4,4]

    def reset(self):
        self.isBeginningOfTrack: bool = True
        self.isDirty: bool = True
        self.currentInstrument: str = ''
        self.currClef: clef = clef.TrebleClef()
        self.currPart: stream.Part = stream.Part()
        # quarterNum = 3: length 3 is a quarter note, length 1 is a triplet
        self.quarterNum: int = 0
        # 3: ###, -4: bbbb
        self.keySigature: int = 0
        self.timeSignature: Tuple[int,int] = [4,4]

    
    def setCurrentInstrument(self, insName: str):
        self.currentInstrument = insName
        self.currPart.partName = insName
    
    def getClef(self):
        return self.currClef
    def setClef(self, clefNum: int):
        match clefNum:
            case 4: self.currClef = clef.TrebleClef()
            case 13: self.currClef = clef.AltoClef()
            case 12: self.currClef = clef.TenorClef()
            case 22: self.currClef = clef.BassClef()
            case _: self.currClef = clef.TrebleClef()
        
    def getCurrPart(self):
        return self.currPart
    
    def getKeySignature(self):
        return self.keySigature
    def setKeySignature(self, ks:int):
        self.keySigature = ks
        
    def setQuarterNum(self, qn:int):
        self.quarterNum = qn
    def getQuarterNum(self):
        return self.quarterNum
        
    def setTimeSignature(self, ts: Tuple[int,int]):
        if ts[1] == 1 and ts[0] == 1:
            self.timeSignature = [4,4]
        elif ts[1]*ts[0] != 0:
            self.timeSignature = ts
    def getTimeSignatureString(self):
        ts = self.timeSignature
        return f'{ts[0]}/{ts[1]}'
    def updateParserState(self, line: str):
        # line is something like '$  K:0   Q:12   T:1/1  C:4   D:Allegro (\\0& = 84)'
        # parse keySignature
        ksMatch = re.search(r"K:(\d+)", line)
        if ksMatch:
            ks = int(ksMatch.group(1))
            self.setKeySignature(ks)
            printtt(f"setting key signature to {ks}")

        # parse quarter Num
        qnMatch = re.search(r"Q:(\d+)", line)
        if qnMatch:
            qn = int(qnMatch.group(1))
            self.setQuarterNum(qn)
            printtt(f"setting quarter number to {qn}")

        tsMatch = re.search(r"T:(\d+)/(\d+)", line)
        if tsMatch:
            num = int(tsMatch.group(1))
            den = int(tsMatch.group(2))
            self.setTimeSignature([num, den])
            printtt(f"setting time signature to {num}/{den}")

        clfMatch = re.search(r"C:(\d+)", line)
        if clfMatch:
            clef_num = int(clfMatch.group(1))
            self.setClef(clef_num)
            printtt(f"setting clef to {clef_num}")
        
        self.isDirty = True
        

    def addMeasureToPart(self, measure: stream.Measure):
        self.currPart.append(measure)

    def getIsDirty(self):
        return self.isDirty
    def setDirtyFalse(self):
        self.isDirty = False


class RestNGObj:
    def __init__(self, pitchs: List[str], dur: int):
        self.pitchs = pitchs
        self.dur = dur
    def decode(self, div: int):
        onlyNote = [p.replace('f','-') for p in self.pitchs if p != 'rest']
        ql = self.dur/div
        if ql == 0:
            print()
        if len(onlyNote) == 0:
            return note.Rest(quarterLength=ql)
        elif len(onlyNote) == 0:
            return note.Note(onlyNote[0], quarterLength=ql)
        else:
            return chord.Chord(onlyNote, quarterLength=ql)

def parseMeasure(msStringLst: List[str], parserState: ParserState, currentMeasureNumber: int):
    m: stream.Measure = stream.Measure(number=currentMeasureNumber)
    msDict = dict()
    currOnset = 0
    prevOnset = 0
    prevLen = 0
    for msString in msStringLst:
        ky = msString[0]
        match ky:
            case 'A'|'B'|'C'|'D'|'E'|'F'|'G'|'r':
                noteStr = msString[0:4].strip()
                dur = int(msString[5:8].strip())
                dictKy = f'{currOnset},{dur}'
                if msDict.get(dictKy):
                    msDict[dictKy].append(noteStr)
                else:
                    msDict[dictKy] = [noteStr]
                prevOnset = currOnset
                prevLen = dur
                currOnset += dur
            case ' ':
                # chord
                dur = int(msString[5:8].strip())
                dictKy = f'{prevOnset},{prevLen}'
                noteStr = msString[1:5].strip()
                msDict[dictKy].append(noteStr)
            case 'b':
                # going back
                dur = int(msString[5:8])
                currOnset -= dur
            case 'i':
                dur = int(msString[5:8])
                currOnset += dur
                # invisible rest, ignore
                # rest
    resLst = []
    while len(msDict) > 0:
        currTrk = []
        prevEnd = 0
        keys_to_delete = []
        consumed_any = False
        for dictKy in list(msDict):  # iterate over a snapshot of keys
            match = re.search(r"(\d+),(\d+)", dictKy)
            if not match:
                keys_to_delete.append(dictKy)
                continue
            onset = int(match.group(1))
            dur = int(match.group(2))
            if onset == prevEnd:
                currTrk.append(RestNGObj(msDict[dictKy], dur))
                keys_to_delete.append(dictKy)
                prevEnd += dur
                consumed_any = True
            elif onset > prevEnd:
                currTrk.append(RestNGObj(['rest'], onset - prevEnd))
                currTrk.append(RestNGObj(msDict[dictKy], dur))
                keys_to_delete.append(dictKy)
                prevEnd = onset + dur
                consumed_any = True
        for k in keys_to_delete:
            del msDict[k]
        resLst.append(currTrk)
        if not consumed_any:  # nothing was consumed → infinite loop guard
            printtt(f"nothing updated in voice, left: {msDict}")
            break
    if parserState.getIsDirty():
        m.append(parserState.getClef())
        m.append(key.KeySignature(parserState.getKeySignature()))
        m.append(meter.TimeSignature(parserState.getTimeSignatureString()))
        parserState.setDirtyFalse()
    # Ef3: flat F#3: sharp
    if len(resLst) == 1:
        # only one track
        for r in resLst[0]:
            m.append(r.decode(parserState.getQuarterNum()))
        return m
    elif len(resLst) > 1:
        for i in range(len(resLst)):
            v = stream.Voice()
            for r in resLst[i]:
                v.append(r.decode(parserState.getQuarterNum()))
            m.insert(0, v)
        return m
    else:
        printtt("issue processing measure")
        return m



def numberToString(num: int, strLen: int = 6):
    stringNum = str(num)
    strAppend = strLen-len(stringNum)
    return ' '*strAppend + stringNum



if __name__ == '__main__':
    md2Paths = [rf"C:\Ellie\APIs\25-omr\md_gt\{j}_{i}.md2" for i in range(1,5) for j in [1,2,3,4,6,7,8]]
    md2Paths.append(rf"C:\Ellie\APIs\25-omr\md_gt\6_5.md2")
    for md2Path in md2Paths:
        score = stream.Score()
        allTrackStringsDict = dict()
        allStrings = []
        with open(md2Path, "r", encoding="utf-8") as f:
            for line_num, lineOrig in enumerate(f, start=1):
                line = lineOrig.rstrip("\n")
                allStrings.append(line)
        parserState = ParserState()
        skipUntilLine = -1
        isComment = False
        measureAccumList = []
        currentMeasureNumber = 1
        inMeasureTitle = ['A','B','C','D','E','F','G',' ','b','i','r']
        ignoreList = ['S','*','g','f','c']
        for (line_num, line) in enumerate(allStrings):
            fmtLineNum = numberToString(line_num+1)
            if (line_num < skipUntilLine 
                or len(line) < 1
                or line[0] == '@' # single line comments
                or line[0] == 'P' # print suggestion record
                or (isComment and line[0] != '&')
                ):
                printt(f"{fmtLineNum}:")
                continue
            line = line.rstrip("\n")
            if line.startswith("/END"):
                score.append(parserState.getCurrPart())
                parserState.reset()
                continue
            elif line.startswith("&&&&&&"):
                skipUntilLine = line_num + 16
                currentMeasureNumber = 1
                parserState.setCurrentInstrument(allStrings[line_num+11])
                printt(f"{fmtLineNum}:&&&&& start of score")
                continue
            elif line[0] == '&':
                isComment = not isComment
                printt(f"{fmtLineNum}: toggle comment to {isComment}")
                continue
            elif line[0] == '$':
                printt(f"{fmtLineNum}: setting state {line}")
                parserState.updateParserState(line)
                continue
            elif line[0] in inMeasureTitle:
                printt(f"{fmtLineNum}: adding {line[0]} to measure")
                measureAccumList.append(line)
                continue
            elif line[0] == 'm':
                # it's the end of the current measure
                newMeasure = parseMeasure(measureAccumList, parserState, currentMeasureNumber)
                parserState.addMeasureToPart(newMeasure)
                measureAccumList = []
                printt(f"{fmtLineNum}: m {currentMeasureNumber}")
                currentMeasureNumber += 1
                continue
            elif line[0] in ignoreList:
                printt(f"{fmtLineNum}: {line[0]}")
                continue
            elif line.startswith('/'):
                printt(f"{fmtLineNum}: {line}")
                continue
        exporter = ScoreExporter(score)
        xmlStr = exporter.parse()
        xmlBytes = tostring(xmlStr, encoding='unicode')
        with open(md2Path.replace('.md2','.musicxml'), 'w', encoding='utf-8') as f:
            f.write(xmlBytes)
        print(f"finish parsing {md2Path}")
    print()

            
