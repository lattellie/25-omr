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

def addValToList(oriList, startActual, endActual, addValue):
    start = startActual-1
    end = endActual
    oriList[start:end] = [x + addValue for x in dataSeperate[start:end]]
    return oriList
if __name__ == '__main__':
    # BEE7
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

    #BEE5
    # dataSeperate = [1, 12, 33, 53, 77, 98, 119, 141, 160, 170, 181, 194, 223, 248, 267, 287, 309, 333, 356, 376,386, 397, 408, 418, 429, 441, 455, 469, 480, 491, 1, 8, 25, 38, 57, 73, 82, 92, 102, 110,
    #                     118, 133, 148, 164, 172, 183, 187, 191, 204, 215, 230, 1, 12, 38, 61, 82, 83, 104, 123, 131,150, 166, 182, 190, 205, 231, 263, 293, 323, 349, 1, 6, 12, 17, 21, 26, 33, 39, 44, 49,
    #                     54, 59, 62, 69, 74, 81, 85, 90, 94, 104, 112, 117, 122, 127, 132, 136, 140, 145, 150, 161,
    #                     186, 207, 214, 219, 224, 228, 234, 240, 247, 252, 257, 262, 268, 272, 279, 285, 294, 300,
    #                     305, 309, 316, 324, 330, 336, 343, 349, 354, 364, 374, 383, 391, 397, 405, 415, 422, 432]
    # dataSeperate = addValToList(dataSeperate, 87, 136, 2)

    # BEE8
    # dataSeperate = [1,8,13,20,26,34,48,64,80,92,98,112, 124,138,151,163,169,175,182,187,193,200,213,219,226,241,255,263,277,289,295,307,317,328,334,345,349,353,357,364, # 1~40 12+1
    #             1, 8,16,24,32,41,48,55,63,71,78,# 41~51
    #             1,6,11,16 , 21,31,38,52, 66, #52~60 53+1 54+1 59+3
    #             1, 8,22,31,42,54,65,75,85,99,112,123,135,148,161,173,183,191,201,211,225,238,244,250,260,274,291,302,312,324,336,348,361,373,383,391,394,401,407,412,417,428,439,449,454,461,474,487 #61~108 62+1
    #             ]
    # dataSeperate = addValToList(dataSeperate, 12, 40, 1)
    # dataSeperate = addValToList(dataSeperate, 53, 60, 1)
    # dataSeperate = addValToList(dataSeperate, 54, 60, 1)
    # dataSeperate = addValToList(dataSeperate, 59, 60, 3)
    # dataSeperate = addValToList(dataSeperate, 62, 108, 1)
    

    symphonyNum = 7
    numberOfMvts = dataSeperate.count(1) # assume all movements starts at a new page
    filePaths = [rf'md_gt\{symphonyNum}_{i}.musicxml' for i in range(1,numberOfMvts+1)]
    outputFolderName = rf'xml_gt\val{symphonyNum}'
    seperatedList = seperateList(dataSeperate) # separate list by measures
    if not os.path.exists(outputFolderName):
        os.mkdir(outputFolderName)
    seperateMeasure(filePaths, numberOfMvts, seperatedList, outputFolderName)