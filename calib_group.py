import numpy as np
# lengths:List[int], types:List[int], targetLength:List[int]

# 1: rest, 0: single, 2: group
# lengths = [4,6,8,8,8]
# lengths = [2,6,8,8,4]
# lengths = [2,8,8,8,4]
# lengths = [2,6,10,8,8]
# lengths = [6,8,8,8]
# types = [2,2,2,1]
# types = [1,2,2,2,1]
# targetLength = 32
# oneUnit = 8

def calib_group(lengths, types, targetLength, oneUnit, ts = (4,4)):
    tsTop = ts[0]
    tsBottom = ts[1]
    numElem = len(lengths)
    tbSize = (numElem, targetLength+1)
    singleCostTb = np.full(tbSize, np.inf)
    sumCostTb = np.full(tbSize, np.inf)
    btTb = np.full(tbSize, -1)

    ng_cost_base = np.full((targetLength+1), np.inf)
    powers_of_2 = [2**i for i in range(0, int(np.log2(targetLength) + 1))]
    ng_cost_base[powers_of_2] = 0
    for i in range(1,int(np.log2(targetLength))):
        curr_2 = 2**i
        # if i = 3, our current: 2**3 = 8, we can multiply it by 1,2(aka 2+3 = 5 = len(powers_of_2)-1)
        fracts = [q for q in range(curr_2+1,curr_2*2,2)]
        selected_idx = []
        for r in range(len(powers_of_2)-i-1):
            selected_idx += [f*(2**r) for f in fracts]
        ng_cost_base[selected_idx] = 3*i
        
    if ts[0]%3 == 0:
        multiply_of_3 = [oneUnit*3*i for i in range(1,ts[0]//3+1)]
        if ts[0]%2 == 0:
            ng_cost_base[multiply_of_3] = 0
        else:
            ng_cost_base = ng_cost_base
            ng_cost_base[multiply_of_3] = 0


    ng_cost_base[[0,1]] = np.max(ng_cost_base[2:])

    # construct singleCostTb
    for i in range(numElem):
        origLen = lengths[i]
        currWeight = 1 if types[i]==2 else 0.5
        if types[i] != 2: # it's not a group
            if origLen in powers_of_2:
                # if length*2^i, make cost i
                for times in range(int(np.log2(targetLength/origLen))+1):
                    singleCostTb[i,origLen*(2**times)] = times*currWeight
                # if length/2^i, make cost i
                for div in range(int(np.log2(origLen))+1):
                    singleCostTb[i, int(origLen/(2**div))] = div*currWeight
                if int(origLen*1.5) == origLen*1.5 and origLen*1.5<targetLength:
                    singleCostTb[i, int(origLen*1.5)] = 1
            elif origLen/1.5 in powers_of_2: #附點
                noDotLen = int(origLen/1.5)
                singleCostTb[i,noDotLen] = 1

                for times in range(int(np.log2(targetLength/origLen))+1-int(noDotLen/targetLength)):
                    singleCostTb[i,int(noDotLen*(2**times)*1.5)] = times*currWeight
                for div in range(int(noDotLen/targetLength),int(np.log2(noDotLen))):
                    # origLen/(2*m*div) will be integer for sure
                    singleCostTb[i, int(origLen/(2**div))] = div*currWeight
            else:
                singleCostTb[i,0] = 0
                # weird ones
                pass
            singleCostTb[i,0] = max(1, singleCostTb[i,1]//2)
        else: # it's a group
            cost_dist = np.full((targetLength+1),np.ceil(targetLength/origLen))
            for j in range(1,int(np.ceil(targetLength/origLen))):
                cost_dist[int(np.round(origLen/(2**j))):int(origLen*(2**j))]-=1
            total_cost = cost_dist+ng_cost_base
            if origLen<targetLength:
                total_cost[origLen] = 0
            singleCostTb[i] = total_cost

    # construct sumCostTb, btTb
    sumCostTb[0,:] = singleCostTb[0,:]
    for i in range(1,numElem):
        for ttLen in range(targetLength+1):
            costs = [singleCostTb[i,ttLen-k]+sumCostTb[i-1,k] for k in range(ttLen+1)]
            additionalCost = 0
            if types[i]==2:
                biggestFactor = ttLen & -ttLen # find the biggest n s.t. 2^n is a factor of ttLen
                biggestFactor = max(1, biggestFactor)
                additionalCost = np.ceil(oneUnit/biggestFactor)-1
            sumCostTb[i,ttLen] = np.min(costs)+additionalCost
            btTb[i,ttLen] = np.argmin(costs)
    totalLens = [None]*numElem
    totalLens[-1] = targetLength
    for i in range(1,numElem):
        totalLens[-i-1] = btTb[-i, totalLens[-i]]
    totalLens = [0]+totalLens
    calibLens = [totalLens[r]-totalLens[r-1] for r in range(1, len(totalLens))]
    assert sum(calibLens) == targetLength
    return calibLens

# if __name__ == '__main__':
#     # (0,0), (0,6), (0,13)
#     lengths = [[8,10,8,4], [8,4,4,12],[24,12,12]]
#     types = [[0,2,0,0], [0,1,1,0],[0,0,2]]
#     targetLength = 36
#     oneUnit = 4
#     ts = (9,8)
#     for id in range(len(lengths)):
#         a = calib_group(lengths[id], types[id], targetLength, oneUnit, ts)
#         print(a)