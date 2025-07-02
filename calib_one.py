import numpy as np
# inputLst = [1,1,1,1,1]
# changeWeight = [1,1,0.5,1,1]
# desiredLength = 4

# inputLst = [4,2,1,1,1]
# changeWeight = [1,1,1,1,1]
# desiredLength = 8

# changeWeight: if it's rest or not trusted: 0.5

def calib_onebar(inputLst, changeWeight, desiredLength):
    numElem = len(inputLst)

    tbSize = (numElem, desiredLength+1)
    singleCostTb = np.full(tbSize, np.inf)
    cummCostTb = np.full(tbSize, np.inf)
    btTb = np.full(tbSize, -1)

    powers_of_2 = [2**i for i in range(0, int(np.log2(desiredLength) + 1))]

    # construct singleCostTb
    for i in range(numElem):
        currLen = inputLst[i]
        currWeight = changeWeight[i]
        if currLen in powers_of_2:
            for times in range(int(np.log2(desiredLength/currLen))+1):
                singleCostTb[i,currLen*(2**times)] = times*currWeight
            for div in range(int(np.log2(currLen))+1):
                singleCostTb[i, int(currLen/(2**div))] = div*currWeight
            if currWeight == 1:
                singleCostTb[i,0] = max(1, singleCostTb[i,1])*4
            else:
                singleCostTb[i,0] = 1
        elif currLen/1.5 in powers_of_2: #附點
            noDotLen = int(currLen/1.5) # currLen: 3x => 2*3x, 4*3x... => 3x/2, 3x/4
            if noDotLen <= desiredLength:
                singleCostTb[i,noDotLen] = 1
            for times in range(int(np.ceil(np.log2(desiredLength/noDotLen)))):
                singleCostTb[i,int(noDotLen*(2**times)*1.5)] = times*currWeight
            for div in range(int(np.log2(noDotLen))):
                # noDotLen/(2**div) will be integer for sure?
                if int(noDotLen/(2**div)) <= desiredLength:
                    singleCostTb[i, int(noDotLen/(2**div))] = div*currWeight
            if currWeight == 1:
                singleCostTb[i,0] = max(1, singleCostTb[i,1])*4
            else:
                singleCostTb[i,0] = 1
        else:
            # only case is when there's a 1/2 rest that was removed
            singleCostTb[i,0] = 0

    # construct cummCostTb, btTb
    cummCostTb[0,:] = singleCostTb[0,:]
    for i in range(1,numElem):
        for ttLen in range(desiredLength+1):
            costs = [singleCostTb[i,ttLen-k]+cummCostTb[i-1,k] for k in range(ttLen+1)]
            cummCostTb[i,ttLen] = np.min(costs)
            btTb[i,ttLen] = np.argmin(costs)

    totalLens = [None]*numElem
    totalLens[-1] = desiredLength
    for i in range(1,numElem):
        totalLens[-i-1] = btTb[-i, totalLens[-i]]
    totalLens = [0]+totalLens
    calibLens = [totalLens[r]-totalLens[r-1] for r in range(1, len(totalLens))]
    assert sum(calibLens) == desiredLength
    return calibLens