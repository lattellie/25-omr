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
        if idx !=0:
            continue
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

def addValToList(oriList, startActual, endActual, addValue):
    start = startActual-1
    end = endActual
    oriList[start:end] = [x + addValue for x in dataSeperate[start:end]]
    return oriList
if __name__ == '__main__':
    dataSeperate = [1,8,16,19,22,30,36,39,46,53,63,75,87,91,95,99,103,108,118,127,138,148,157,162,167,171,184,196,205,209,213,225,235,245,255,260,270,279,283,287,291,295,300,311,323,331,335,346,358,366,370,376,380,384,397,409,416,422,428,433,439,444, #1~62
                    1,32,63,77,84,91,106,120,134,146,158,170,182,192,212,217,230,244,261, #63~81
                    1,16,34,56,76,97,118,135,143,160,181,198,214,233,250,259,277,293,317,337,357,377,396,404,424,441,459,477,496,513,531,554,574,593,613,629,637,645, #82~119 # 84 + 1, 91 + 4+1, 96~+1 111+1
                    1,7,12,17,21,28,34,39,52,60,73,81,90,104,113,119,122,129,146,151,154,159,163,177,191,203,218,226,231,238,245,251,257,263,271,278,290,302,314,320,328,335,340,347,358,370,382,394,400,406,414,421,429,437,445,453,459 #148+1 #120~ 122+1 124+1 136+5 140+2 142+1
                    ]
    #82~119 # 84 + 1, 91 + 4+1, 96~+1 111+1
    dataSeperate = addValToList(dataSeperate, 91,119,4)
    #120-176 148+1 #120~ 122+1 124+1 136+5 140+2 142+1
    dataSeperate = addValToList(dataSeperate, 122, 176,1)
    dataSeperate = addValToList(dataSeperate, 124, 176,1)
    dataSeperate = addValToList(dataSeperate, 136, 176,5)
    dataSeperate = addValToList(dataSeperate, 140, 176,2)
    dataSeperate = addValToList(dataSeperate, 142, 176,1)
    dataSeperate = addValToList(dataSeperate, 148, 176,1)
    


    symphonyNum = 7
    numberOfMvts = dataSeperate.count(1) # assume all movements starts at a new page
    filePaths = [rf'xml_gt\{symphonyNum}_{i}.xml' for i in range(1,numberOfMvts+1)]
    outputFolderName = rf'orch_dataset\Bee_{symphonyNum}_challenge\val'
    seperatedList = seperateList(dataSeperate) # separate list by measures
    if not os.path.exists(outputFolderName):
        os.mkdir(outputFolderName)
    seperateMeasure(filePaths, numberOfMvts, seperatedList, outputFolderName)