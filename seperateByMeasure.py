from music21 import converter, stream
import os
import copy

def slice_score_clean(score, start, end):
    new_score = stream.Score()

    for part in score.parts:
        new_part = stream.Part()
        new_part.insert(0, part.getInstrument())

        for m in part.measures(start, end):
            new_part.append(copy.deepcopy(m)) 

        new_score.append(new_part)

    return new_score

def seperateMeasure(filePaths, numberOfMvts, beginningIndexes, outputFolder):
    assert len(filePaths) == numberOfMvts
    assert len(beginningIndexes) == numberOfMvts
    for idx in range(numberOfMvts):
        filePath = filePaths[idx]
        beginningIndex = beginningIndexes[idx]
        score = converter.parse(filePath)
        for i in range(len(beginningIndex) - 1):
            start = beginningIndex[i]
            end = beginningIndex[i + 1] - 1  # stop before next boundary
            segment = slice_score_clean(score, start, end)
            outputFileName = ''
            if '.musicxml' in filePath:
                outputFileName = f"{os.path.basename(filePath).replace('.musicxml',f'_{i+1}.musicxml')}"
            elif '.xml' in filePath:
                outputFileName = f"{os.path.basename(filePath).replace('.xml',f'_{i+1}.xml')}"
            outputPath = os.path.join(outputFolder, outputFileName)
            segment.write('musicxml', outputPath)
            print(f"saved score to {outputPath}")
        start = beginningIndex[-1]
        end = score.measures(start, None)
        segment = slice_score_clean(score, start, None)
        outputFileName = ''
        if '.musicxml' in filePath:
            outputFileName = f"{os.path.basename(filePath).replace('.musicxml',f'_{i+2}.musicxml')}"
        elif '.xml' in filePath:
            outputFileName = f"{os.path.basename(filePath).replace('.xml',f'_{i+2}.xml')}"
        outputPath = os.path.join(outputFolder, outputFileName)
        segment.write('musicxml', outputPath)
        print(f"saved score to {outputPath}")
def seperateList(measureLst):
    measureNumberLists = []
    current = []
    for x in measureLst:
        if x == 1:
            if current: 
                measureNumberLists.append(current)
            current = [1]
        elif current: 
            current.append(x)
    if current:
        measureNumberLists.append(current)
    return measureNumberLists

if __name__ == '__main__':
    dataSeperate = [1, 12, 33, 53, 77, 98, 119, 141, 160, 170, 181, 194, 223, 248, 267, 287, 309, 333, 356, 376,
                    386, 397, 408, 418, 429, 441, 455, 469, 480, 491, 1, 8, 25, 38, 57, 73, 82, 92, 102, 110,
                    118, 133, 148, 164, 172, 183, 187, 191, 204, 215, 230, 1, 12, 38, 61, 82, 83, 104, 123, 131,
                    150, 166, 182, 190, 205, 231, 263, 293, 323, 349, 1, 6, 12, 17, 21, 26, 33, 39, 44, 49,
                    54, 59, 62, 69, 74, 81, 85, 90, 94, 104, 112, 117, 122, 127, 132, 136, 140, 145, 150, 161,
                    186, 207, 214, 219, 224, 228, 234, 240, 247, 252, 257, 262, 268, 272, 279, 285, 294, 300,
                    305, 309, 316, 324, 330, 336, 343, 349, 354, 364, 374, 383, 391, 397, 405, 415, 422, 432]
    symphonyNum = 5
    numberOfMvts = dataSeperate.count(1) # assume all movements starts at a new page
    filePaths = [rf'xml_gt\{symphonyNum}_{i}.xml' for i in range(1,numberOfMvts+1)]
    outputFolderName = rf'orch_dataset\Bee_{symphonyNum}_challenge\val'
    seperatedList = seperateList(dataSeperate) # separate list by measures
    if not os.path.exists(outputFolderName):
        os.mkdir(outputFolderName)
    seperateMeasure(filePaths, numberOfMvts, seperatedList, outputFolderName)