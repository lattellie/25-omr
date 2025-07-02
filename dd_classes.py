from fractions import Fraction
from typing import Tuple

class RestNg:
    def __init__(self, length, groupId, isRest, numNotes, beamLength, beamEnd):
        self.length:Fraction = length
        self.groupId:int = groupId # =1: rest, 0: singleNote, other: groupId
        self.isRest:bool = isRest
        self.numNotes:int  = numNotes # number of notes in the Ng, 0 for rest
        self.beamLength:Tuple[int,int] = beamLength # single note in Ng: (left, right) else (-1, -1) for rest, others
        self.beamEnd: Tuple[int,int]= beamEnd # single note in Ng: the x,y for the end of beam or (-1,-1)
        self.newGroupId: int|None = None        
        self.newLength:Fraction|None = None # the tuned length
