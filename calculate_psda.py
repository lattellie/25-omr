import csv
from music21 import converter, note, chord, stream
import os
from typing import List

from music21.musicxml.m21ToXml import ScoreExporter
from xml.etree.ElementTree import tostring

dataSepList = dict()
dataSepList['bee5'] = [1, 12, 33, 53, 77, 98, 119, 141, 160, 170, 181, 194, 223, 248, 267, 287, 309, 333, 356, 376,386, 397, 408, 418, 429, 441, 455, 469, 480, 491, 
                        1, 8, 25, 38, 57, 73, 82, 92, 102, 110, 118, 133, 148, 164, 172, 183, 187, 191, 204, 215, 230, 
                        1, 12, 38, 61, 82, 83, 104, 123, 131,150, 166, 182, 190, 205, 231, 263, 293, 323, 349, 
                        1, 6, 12, 17, 21, 26, 33, 39, 44, 49, 54, 59, 62, 69, 74, 81, 85, 90, 94, 104, 112, 117, 122, 127, 132, 136, 140, 145, 150, 161, 186, 207, 214, 219, 224, 228, 234, 240, 247, 252, 257, 262, 268, 272, 279, 285, 294, 300, 305, 309, 316, 324, 330, 336, 343, 349, 354, 364, 374, 383, 391, 397, 405, 415, 422, 432]
dataSepList['bee6'] = [1,17,31,43,57,83,95,111,124,139,154,166,177,196,209,221,241,260,276,298,317,329,354,368,381,395,407,426,440,452,464,479,493,
                        1,4,8,10,12,16,21,25,29,31,35,39,42,44,46,49,51,53,57,61,65,70,74,79,83,87,89,91,93,95,97,99,101,103,107,111,116,118,121,123,125,127,133,
                        1,19,39,59,80,111,141,161,176,191,205,224,244,125,
                        1,14,22,25,28,31,34,40,17,51,56,65,72,78,82,87,93,103,107,110,114,125,137,145,
                        1,14,23,26,32,36,40,44,48,52,57,28,77,85,93,101,105,109,117,131,134,137,141,149,153,157,161,170,179,191,196,204,212,219,225,231,245,260]
dataSepList['bee7'] = [1,8,16,19,22,30,36,39,46,53,63,75,87,91,95,99,103,108,118,127,138,148,157,162,167,171,184,196,205,209,213,225,235,245,255,260,270,279,283,287,291,295,300,311,323,331,335,346,358,366,370,376,380,384,397,409,416,422,428,433,439,444,
                        1,32,63,77,84,91,106,120,134,146,158,170,182,192,212,217,230,244,261,
                        1,16,34,56,76,97,118,135,143,160,181,198,214,233,250,259,277,293,317,337,357,377,396,404,424,441,459,477,496,513,531,554,574,593,613,629,637,645,
                        1,7,12,17,21,28,34,39,52,60,73,81,90,104,113,119,122,129,146,151,154,159,163,177,191,203,218,226,231,238,245,251,257,263,271,278,290,302,314,320,328,335,340,347,358,370,382,394,400,406,414,421,429,437,445,453,459]
dataSepList['bee8'] = [1,8,13,20,26,34,48,64,80,92,98,112,124,138,151,163,169,175,182,187,193,200,213,219,226,241,255,263,277,289,295,307,317,328,334,345,349,353,357,364,
                        1,8,16,24,32,41,48,55,63,71,78,
                        1,6,11,16,21,31,38,52,66,
                        1,8,22,31,42,54,65,75,85,99,112,123,135,148,161,173,183,191,201,211,225,238,244,250,260,274,291,302,312,324,336,348,361,373,383,391,394,401,407,412,417,428,439,449,454,461,474,487]

def lcs(seq1, seq2):
    n, m = len(seq1), len(seq2)

    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Fill table
    for i in range(n):
        for j in range(m):
            if seq1[i] == seq2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    print("finish creating table of size")
    # Reconstruct LCS
    i, j = n, m
    lcs_seq = []

    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            lcs_seq.append(seq1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return lcs_seq[::-1]  # reverse

def musicxml_to_csv(xml_path, csv_path, trackLength):
    score = converter.parse(xml_path)

    rows = []

    # Iterate over parts (tracks)
    for track_idx, part in enumerate(score.parts):
        if track_idx >= trackLength:
            continue
        # Go measure by measure
        for m_idx, measure in enumerate(part.getElementsByClass('Measure')):
            bar_number = measure.number

            # Iterate over notes/rests/chords in the measure
            for elem in measure.notesAndRests:
                onset = elem.offset  # relative to measure start
                duration = elem.quarterLength

                if isinstance(elem, note.Note):
                    pitch = elem.pitch.nameWithOctave

                    rows.append([
                        track_idx,
                        bar_number,
                        onset,
                        duration,
                        pitch
                    ])

                elif isinstance(elem, note.Rest):
                    rows.append([
                        track_idx,
                        bar_number,
                        onset,
                        duration,
                        "Rest"
                    ])

                elif isinstance(elem, chord.Chord):
                    # If chord, output one row per note
                    for p in elem.pitches:
                        rows.append([
                            track_idx,
                            bar_number,
                            onset,
                            duration,
                            p.nameWithOctave
                        ])

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track", "bar", "onset", "duration", "pitch"])
        writer.writerows(rows)

    return len(score.parts)

# psda, psd, pda, pd, pa, p
def csv_to_string_list(csv_path):
    def convertPitch(pitch:str):
        if len(pitch) == 3:
            return pitch[0]+pitch[2]
        else:
            return pitch
    psda = []
    psd = []
    pda = []
    pd = []
    pa = []
    p = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)

        for row in reader:
            track = 0 # row["track"]
            onset = row["onset"]
            duration = row["duration"]
            pitch = row["pitch"]

            s = f"{track}_{pitch}_{onset}_{duration}"
            psda.append(s)
            s = f"{track}_{convertPitch(pitch)}_{onset}_{duration}"
            psd.append(s)
            s = f"{track}_{pitch}_{duration}"
            pda.append(s)
            s = f"{track}_{convertPitch(pitch)}_{duration}"
            pd.append(s)
            s = f"{track}_{pitch}"
            pa.append(s)
            s = f"{track}_{convertPitch(pitch)}"
            p.append(s)
        ret = dict()
        ret['psda'] = psda
        ret['psd'] = psd
        ret['pda'] = pda
        ret['pd'] = pd
        ret['pa'] = pa
        ret['p'] = p
    return ret
def get_idx_of_ones(lst):
    return [i for i, x in enumerate(lst) if x == 1]

def get_val_name(startOfMvt, currPgNum):
    mvtNum = 0
    pageNum = 0
    for i in range(len(startOfMvt)):
        if startOfMvt[i] < currPgNum:
            mvtNum = i+1
            pageNum = currPgNum-startOfMvt[i]
    return f"{mvtNum}_{pageNum}"



def combineXMLs(xmlList: List[str]):
    scores = [converter.parse(xml) for xml in xmlList]
    combined = stream.Score()

    for parts in zip(*[score.parts for score in scores]):
        combined_part = stream.Part()
        combined_part.id = parts[0].id
        combined_part.partName = parts[0].partName

        for part in parts:
            for measure in part.getElementsByClass(stream.Measure):
                combined_part.append(measure)

        combined.append(combined_part)
    return combined
    
def lcsByPage(symphonyNum: int):
    rootValFolder = rf"C:\Ellie\APIs\25-omr\xml_gt\val{symphonyNum}"
    rootXmlFolder = rf"C:\Ellie\APIs\25-omr\gt_pred\bee{symphonyNum}_xmls"
    outputFolder = rf"C:\Ellie\APIs\25-omr\xml_gt\csv{symphonyNum}"
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    outputOutFolder = os.path.join(outputFolder, 'out')
    outputValFolder = os.path.join(outputFolder, 'val')
    if not os.path.exists(outputOutFolder):
        os.mkdir(outputOutFolder)
    if not os.path.exists(outputValFolder):
        os.mkdir(outputValFolder)
    dataSeperate = dataSepList.get(f'bee{symphonyNum}')
    avgCsvPath = os.path.join(rf"C:\Ellie\APIs\25-omr\xml_gt\bee{symphonyNum}_psda.csv")
    rootAvgCsPath = avgCsvPath
    cnt = 0
    while os.path.exists(avgCsvPath):
        avgCsvPath = rootAvgCsPath.replace('.csv',f'_{cnt}.csv')
        cnt += 1
    for i in range(1, len(dataSeperate)+1):
        mvtPg = get_val_name(get_idx_of_ones(dataSeperate), i)
        valXmlPath = os.path.join(rootValFolder, rf'{symphonyNum}_{mvtPg}.musicxml')
        xmlXmlPath = os.path.join(rootXmlFolder, rf'Bee_{symphonyNum}_challenge_{i}.musicxml')

        xmlScore = converter.parse(xmlXmlPath)
        valScore = converter.parse(valXmlPath)

        minNumTracks = min(len(xmlScore.parts), len(valScore.parts))
        # minNumMeasures = min(len(xmlScore.parts[0].getElementsByClass('Measure')),len(valScore.parts[0].getElementsByClass('Measure')))
        
        xmlCsvPath = os.path.join(outputOutFolder, rf'out{i}.csv')
        valCsvPath = os.path.join(outputValFolder, rf'truth{i}.csv')
        musicxml_to_csv(xmlXmlPath, xmlCsvPath, minNumTracks)
        musicxml_to_csv(valXmlPath, valCsvPath, minNumTracks)

        xmlRet = csv_to_string_list(xmlCsvPath)
        valRet = csv_to_string_list(valCsvPath)
        print()
        metricsName = ['psda','psd','pda','pd','pa','p']
        rowNames = ["lenPred","lenTruth"]
        for mt in metricsName:
            # /ground truth 命中多少：precision
            # /our length 正確多少：recall
            rowNames += [f"{mt}_length", f"{mt}/pred", f"{mt}/gt"]
        numCorrect = dict() # numCorrect['psda'] = [len(lcs), len(xml), len(val), correctOfXml, correctOfVal]
        lenXml = len(xmlRet[metricsName[0]])
        lenVal = len(valRet[metricsName[0]])
        rowResToWrite = [lenXml, lenVal]

        for ky in metricsName:
            lcsRes = len(lcs(xmlRet[ky], valRet[ky]))
            numCorrect[ky] = [lcsRes, lenXml, lenVal, lcsRes/lenXml, lcsRes/lenVal]
            rowResToWrite += [lcsRes, lcsRes/lenXml, lcsRes/lenVal]
            print(f"{ky}: {numCorrect[ky]}")
        with open(avgCsvPath, "a", newline="") as f:
            writer = csv.writer(f)
            if not os.path.isfile(avgCsvPath):
                writer.writerow(rowNames)
            writer.writerow(rowResToWrite)
        print(f"page{i}: {rowResToWrite}")
    print()

def lcsByMvt(syphonyNum: int):
    rootValFolder = rf"C:\Ellie\APIs\25-omr\xml_gt\val{symphonyNum}"
    rootXmlFolder = rf"C:\Ellie\APIs\25-omr\gt_pred\bee{symphonyNum}_xmls"
    rootGtFolder = rf"C:\Ellie\APIs\25-omr\md_gt"
    rootRootCsvFolder = rf"C:\Ellie\APIs\25-omr\xml_gt\csv{symphonyNum}"
    rootCsvFolder = os.path.join(rootRootCsvFolder, "mvt")
    checkAndCreateFolder([rootValFolder, rootXmlFolder, rootRootCsvFolder, rootCsvFolder])
    dataSeperate = dataSepList.get(f'bee{symphonyNum}')
    outputFolder = 'xml_gt'
    beginningOfMvts = [i+1 for i in range(len(dataSeperate)) if dataSeperate[i] == 1]
    numMvts = len(beginningOfMvts)
    beginningOfMvts.append(len(dataSeperate)+1)
    for i in range(numMvts):
        mvtNum = i+1
        start = beginningOfMvts[i]
        end = beginningOfMvts[i+1] # should not be included
        xmlPaths = [os.path.join(rootXmlFolder,f"Bee_{symphonyNum}_challenge_{f}.musicxml") for f in range(start,end)]
        score = combineXMLs(xmlPaths)
        outputPath = os.path.join(outputFolder, f"bee{symphonyNum}_{mvtNum}.musicxml")
        avgCsvPath = os.path.join(rootCsvFolder, f"bee_{symphonyNum}_{mvtNum}")
        if not os.path.exists(outputPath):
            exporter = ScoreExporter(score)
            xmlStr = exporter.parse()
            xmlBytes = tostring(xmlStr, encoding='unicode')
            with open(outputPath, 'w', encoding='utf-8') as f:
                f.write(xmlBytes)
            print(f"finish parsing {outputPath}")
        valXmlPath = os.path.join(rootGtFolder, rf'{symphonyNum}_{mvtNum}.musicxml')

        xmlScore = score
        valScore = converter.parse(valXmlPath)

        minNumTracks = min(len(xmlScore.parts), len(valScore.parts))
        # minNumMeasures = min(len(xmlScore.parts[0].getElementsByClass('Measure')),len(valScore.parts[0].getElementsByClass('Measure')))
        
        xmlCsvPath = os.path.join(rootCsvFolder, rf'out_{symphonyNum}_{mvtNum}.csv')
        valCsvPath = os.path.join(rootCsvFolder, rf'truth_{symphonyNum}_{mvtNum}.csv')
        musicxml_to_csv(outputPath, xmlCsvPath, minNumTracks)
        musicxml_to_csv(valXmlPath, valCsvPath, minNumTracks)

        xmlRet = csv_to_string_list(xmlCsvPath)
        valRet = csv_to_string_list(valCsvPath)
        metricsName = ['psda','psd','pda','pd','pa','p']
        rowNames = ["lenPred","lenTruth"]
        for mt in metricsName:
            # /ground truth 命中多少：precision
            # /our length 正確多少：recall
            rowNames += [f"{mt}_length", f"{mt}/pred", f"{mt}/gt"]
        numCorrect = dict() # numCorrect['psda'] = [len(lcs), len(xml), len(val), correctOfXml, correctOfVal]
        lenXml = len(xmlRet[metricsName[0]])
        lenVal = len(valRet[metricsName[0]])
        rowResToWrite = [lenXml, lenVal]

        for ky in metricsName:
            lcsRes = len(lcs(xmlRet[ky], valRet[ky]))
            numCorrect[ky] = [lcsRes, lenXml, lenVal, lcsRes/lenXml, lcsRes/lenVal]
            rowResToWrite += [lcsRes, lcsRes/lenXml, lcsRes/lenVal]
            print(f"{ky}: {numCorrect[ky]}")
        with open(avgCsvPath, "a", newline="") as f:
            writer = csv.writer(f)
            if not os.path.isfile(avgCsvPath):
                writer.writerow(rowNames)
            writer.writerow(rowResToWrite)
        print(f"page{i}: {rowResToWrite}")

def checkAndCreateFolder(folderPaths):
    for folderPath in folderPaths:
        if not os.path.exists(folderPath):
            os.mkdir(folderPath)
def musicxml_to_csv_single_track(xml_path, csv_path, track_idx):
    score = converter.parse(xml_path)
    rows = []
    part = score.parts[track_idx]

    for measure in part.getElementsByClass('Measure'):
        bar_number = measure.number
        for elem in measure.notesAndRests:
            onset = elem.offset
            duration = elem.quarterLength

            if isinstance(elem, note.Note):
                rows.append([0, bar_number, onset, duration, elem.pitch.nameWithOctave])
            elif isinstance(elem, note.Rest):
                rows.append([0, bar_number, onset, duration, "Rest"])
            elif isinstance(elem, chord.Chord):
                for p in elem.pitches:
                    rows.append([0, bar_number, onset, duration, p.nameWithOctave])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track", "bar", "onset", "duration", "pitch"])
        writer.writerows(rows)

def lcsByMvtPart(symphonyNum: int):
    rootValFolder = rf"C:\Ellie\APIs\25-omr\xml_gt\val{symphonyNum}"
    rootXmlFolder = rf"C:\Ellie\APIs\25-omr\gt_pred\bee{symphonyNum}_xmls"
    rootGtFolder = rf"C:\Ellie\APIs\25-omr\md_gt"
    rootRootCsvFolder = rf"C:\Ellie\APIs\25-omr\xml_gt\csv{symphonyNum}"
    rootCsvFolder = os.path.join(rootRootCsvFolder, "mvt")
    checkAndCreateFolder([rootValFolder, rootXmlFolder, rootRootCsvFolder, rootCsvFolder])
    dataSeperate = dataSepList.get(f'bee{symphonyNum}')
    outputFolder = 'xml_gt'
    beginningOfMvts = [i+1 for i in range(len(dataSeperate)) if dataSeperate[i] == 1]
    numMvts = len(beginningOfMvts)
    beginningOfMvts.append(len(dataSeperate)+1)

    for i in range(numMvts):
        mvtNum = i+1
        start = beginningOfMvts[i]
        end = beginningOfMvts[i+1]
        xmlPaths = [os.path.join(rootXmlFolder, f"Bee_{symphonyNum}_challenge_{f}.musicxml") for f in range(start, end)]
        score = combineXMLs(xmlPaths)
        outputPath = os.path.join(outputFolder, f"bee{symphonyNum}_{mvtNum}.musicxml")
        
        if not os.path.exists(outputPath):
            exporter = ScoreExporter(score)
            xmlStr = exporter.parse()
            xmlBytes = tostring(xmlStr, encoding='unicode')
            with open(outputPath, 'w', encoding='utf-8') as f:
                f.write(xmlBytes)
            print(f"finish parsing {outputPath}")

        valXmlPath = os.path.join(rootGtFolder, rf'{symphonyNum}_{mvtNum}.musicxml')
        xmlScore = score
        valScore = converter.parse(valXmlPath)
        minNumTracks = min(len(xmlScore.parts), len(valScore.parts))

        metricsName = ['psda', 'psd', 'pda', 'pd', 'pa', 'p']
        rowNames = ["part", "lenPred", "lenTruth"]
        for mt in metricsName:
            rowNames += [f"{mt}_length", f"{mt}/pred", f"{mt}/gt"]

        avgCsvPath = os.path.join(rootCsvFolder, f"bee_{symphonyNum}_{mvtNum}.csv")

        with open(avgCsvPath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(rowNames)

        # --- Run LCS per part instead of on the whole movement ---
        for trackIdx in range(minNumTracks):
            xmlCsvPath = os.path.join(rootCsvFolder, rf'out_{symphonyNum}_{mvtNum}_part{trackIdx}.csv')
            valCsvPath = os.path.join(rootCsvFolder, rf'truth_{symphonyNum}_{mvtNum}_part{trackIdx}.csv')

            # Pass trackLength as a single-track window: only export track trackIdx
            musicxml_to_csv_single_track(outputPath, xmlCsvPath, trackIdx)
            musicxml_to_csv_single_track(valXmlPath, valCsvPath, trackIdx)

            xmlRet = csv_to_string_list(xmlCsvPath)
            valRet = csv_to_string_list(valCsvPath)

            lenXml = len(xmlRet[metricsName[0]])
            lenVal = len(valRet[metricsName[0]])
            rowResToWrite = [trackIdx, lenXml, lenVal]

            for ky in metricsName:
                lcsRes = len(lcs(xmlRet[ky], valRet[ky]))
                rowResToWrite += [lcsRes, lcsRes/lenXml, lcsRes/lenVal]
                print(f"  part{trackIdx} {ky}: lcs={lcsRes}, pred={lenXml}, gt={lenVal}")

            with open(avgCsvPath, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(rowResToWrite)

            print(f"mvt{mvtNum} part{trackIdx}: {rowResToWrite}")
if __name__ == '__main__':
    symphonyNum = 7
    # lcsByPage()
    # lcsByMvt(symphonyNum)
    lcsByMvtPart(symphonyNum)
  

