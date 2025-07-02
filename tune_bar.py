from dd_classes import RestNg
from typing import List,Tuple, Union, Set
from fractions import Fraction
import numpy as np
import math
from calib_group import calib_group
from calib_one import calib_onebar


def calibOneGroup(elmList:List[RestNg], desiredLength:Fraction):
    if desiredLength == 0:
        for elm in elmList:
            elm.newLength = 0
        return
    realLens = []
    realLens.append(elmList[0].beamLength[1])
    for i in range(1,len(elmList)-1):
        prevBeamLength = elmList[i-1].beamLength
        currBeamLength = elmList[i].beamLength
        nextBeamLength = elmList[i+1].beamLength
        if abs(currBeamLength[1]-currBeamLength[0])<= currBeamLength[0]/4:
            realLens.append(max(currBeamLength)) # if approx the same, get the longer
        # else it should be either 1) same as prev 2) same as next
        # else it is probably an error and has to be changed 
        elif abs(prevBeamLength[1]-currBeamLength[0])<=currBeamLength[0]/4 or abs(nextBeamLength[0]-currBeamLength[1])<= currBeamLength[0]/4:
            realLens.append(max(currBeamLength))
        else:
            realLens.append(0)
    realLens.append(elmList[-1].beamLength[0])
    stemYs = [e.beamEnd[1] for e in elmList]
    stemXs = [e.beamEnd[0] for e in elmList]
    endBeamTrust = [True]*len(elmList)
    if -1 in stemYs or -1 in stemXs:
        endBeamTrust = [False]*len(elmList)
    else:
        stemAtan = [math.atan((stemYs[i]-stemYs[i-1])/(stemXs[i]-stemXs[i-1])) for i in range(1,len(stemXs))]
        irregThresh = math.pi/18
        if len(elmList)<=2:
            endBeamTrust = [False, False]
        elif len(elmList) == 3:
            if abs(stemAtan[1]-stemAtan[0])>irregThresh:
                endBeamTrust = [False]*3
        else:
            medTan = np.median(stemAtan)
            endBeamTrust = [abs(stemAtan[i]-medTan)<irregThresh for i in range(len(stemAtan))]
    # case one: obvious nth tuplets
    if False not in endBeamTrust and False not in [abs(realLens[i]-np.mean(realLens))<max(3, np.mean(realLens)/4) for i in range(len(realLens))]:
        for elm in elmList:
            elm.newLength = desiredLength/len(elmList)
    elif len(np.unique([e.length for e in elmList])) == 1:
        for elm in elmList:
            elm.newLength = desiredLength/len(elmList)
    else:
        changeWeight = [0.5 if e.isRest else 1 for e in elmList]
        if True in endBeamTrust:
            endBeamTrustExtended = [True] + endBeamTrust + [True]
            trust = [endBeamTrustExtended[i] & endBeamTrustExtended[i+1] for i in range(len(endBeamTrustExtended)-1)]
            changeWeight = [1 if i else 0.5 for i in trust]
        denom = max([e.length.denominator for e in elmList]+[32])
        inputLst = [int(e.length*denom) for e in elmList]
        newLen = calib_onebar(inputLst, changeWeight, int(desiredLength*denom))
        for idx, elm in enumerate(elmList):
            elm.newLength = Fraction(newLen[idx]/denom)

def tuneBar(elmList: List[RestNg], targetLength: Tuple[int,int])->List[Fraction]:
    targetLengthFraction = Fraction(targetLength[0], targetLength[1])
    totalLength = Fraction(0,1)
    currGroup = None
    grpLst = [e.groupId for e in elmList] # for grouping the rest inside a horizontal grouping
    uniqId = np.unique(np.array(grpLst)).tolist()
    for uuId in uniqId:
        if uuId>0:
            idIndex = np.where(np.array(grpLst)==uuId)
            if len(idIndex[0])>0:
                for i in range(np.min(idIndex),np.max(idIndex)):
                    grpLst[i] = uuId
    numGroups = 0
    numGroupedNotes = 0
    numSingleNotes = 0
    numSingleRests = 0
    lenNotes = Fraction(0,1)
    lenRests = Fraction(0,1)
    lenSingle = 0
    for idxx, elm in enumerate(elmList):
        elm.newGroupId = grpLst[idxx]
        totalLength += elm.length
        if elm.newGroupId>0: #noteGroup
            if elm.newGroupId != currGroup:
                currGroup = elm.newGroupId
                numGroups+=1
            numGroupedNotes += 1
            if elm.isRest:
                lenRests += elm.length
                if elm.length==Fraction(1,2):
                    print("reassigning element length to smallest")
                    elm.newLength = 0
                    elm.length = 0
            else:
                lenNotes += elm.length
        elif elm.newGroupId == -1: # rest
            numSingleRests+=1
            lenSingle += elm.length
            lenRests += elm.length
        elif elm.newGroupId == 0: # single note
            numSingleNotes+=1
            lenSingle += elm.length
            lenNotes += elm.length
        else:
            raise ValueError("invalid element id")

    # most common type: each notegroup is one beat --> length sums up right already
    if numGroups == targetLength[0] and numSingleNotes == 0 and numSingleRests == 0:
        currElemList:List[RestNg] = []
        currGroup = None
        for elm in elmList:
            if elm.newGroupId != currGroup:
                if len(currElemList) != 0:
                    calibOneGroup(currElemList, Fraction(1,targetLength[1]))
                currGroup = elm.newGroupId
                currElemList = []
                currElemList.append(elm)
            else:
                currElemList.append(elm)
        if len(currElemList) != 0:
            calibOneGroup(currElemList, Fraction(1,targetLength[1]))
    # case: only noteGroup, notegroup length 能整除, nothing else: --> length sums up right already
    elif numGroups != 0 and (targetLength[0]/numGroups)%1==0 and numSingleNotes == 0 and numSingleRests == 0:
        currElemList:List[RestNg] = []
        currGroup = None
        for elm in elmList:
            if elm.newGroupId != currGroup:
                if len(currElemList) > 1:
                    calibOneGroup(currElemList, Fraction(targetLength[0]//numGroups,targetLength[1]))
                currGroup = elm.newGroupId
                currElemList = []
                currElemList.append(elm)
            else:
                currElemList.append(elm)
        if len(currElemList) > 1:
            calibOneGroup(currElemList, Fraction(targetLength[0]//numGroups,targetLength[1]))
    # only has rest
    # or only has group with 1 element (probably small notes) and or rest
    elif ((numGroups == 0 or numGroupedNotes == numGroups) and numSingleNotes == 0): 
        elmList[0].newLength = Fraction(targetLength[0],targetLength[1])
        elmList[0].isRest = True
        for elm in elmList[1:]:
            elm.newLength = 0
    # if everything is fine without rests (misclassified rests in group)
    elif lenNotes == Fraction(targetLength[0], targetLength[1]):
        for elm in elmList:
            if elm.isRest:
                elm.newLength = 0
    elif numGroups < targetLength[0] and lenSingle + Fraction(numGroups,targetLength[1])==targetLengthFraction:
        currElemList:List[RestNg] = []
        currGroup = None
        for elm in elmList:
            if elm.groupId <= 0:
                elm.newLength = elm.length
            elif elm.newGroupId != currGroup:
                if len(currElemList) > 1:
                    calibOneGroup(currElemList, Fraction(1,targetLength[1]))
                currGroup = elm.newGroupId
                currElemList = []
                currElemList.append(elm)
            else:
                currElemList.append(elm)
        if len(currElemList) > 1:
            calibOneGroup(currElemList, Fraction(1,targetLength[1]))
    else:
        newGroupIds = []
        newGroupLengths = []
        newGroupType = [] # 1: rest, 0: single, 2: group
        prevId = -2
        groupMap:List[List[RestNg]] = []
        restNextIsNg = [j for j in range(len(newGroupType) - 1) if newGroupType[j:j+2] == [1,2]]
        denom = max(max([e.length.denominator for e in elmList]), 32)
        for idxx, elm in enumerate(elmList):
            if elm.newGroupId<=0: # is rest (newGroupId = -1)
                newGroupType.append(-elm.newGroupId)
                newGroupLengths.append(int(elm.length*denom))
                newGroupIds.append(elm.newGroupId)
                groupMap.append([elm])
            elif elm.newGroupId != prevId:
                newGroupType.append(2) # is a group
                newGroupLengths.append(int(elm.length*denom))
                newGroupIds.append(elm.newGroupId)
                prevId = elm.newGroupId
                groupMap.append([elm])
            else:
                groupMap[-1].append(elm)
                newGroupLengths[-1] += int(elm.length*denom)
        
        returnedLens = calib_group(newGroupLengths, newGroupType, int(targetLengthFraction*denom), int(denom/targetLength[1]), ts = targetLength)
        for ii, newll in enumerate(returnedLens):
            if sum([elm.length for elm in groupMap[ii]]) == Fraction(newll, denom): # if it's the same as calib-ed length
                continue
            if len(groupMap[ii]) == 1: # if there's only one element in that group -> assign the length
                groupMap[ii][0].newLength = Fraction(newll, denom)
            else: # calib to specific length
                calibOneGroup(groupMap[ii], Fraction(newll, denom))
        # TODO: potentially group rests into noteGroups and redo the mapping?
        
    return [e.newLength for e in elmList]