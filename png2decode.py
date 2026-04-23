

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import statistics

from sklearn.cluster import KMeans
from collections import Counter
from scipy.ndimage import rotate

from exportXml import exportXML
from get_prediction_singleStem import Single_Stem_Classifier
from get_prediction_rest import Rest_Classifier
from get_prediction import Sfn_Clef_classifier

from utils import *
from omr.part1 import runModel1
from omr.staffline_extraction import staff_extract_staffobj
from omr.part2 import runModel2
from omr.bbox import merge_nearby_bbox

STEM_UP_MODEL = Single_Stem_Classifier("training/stemupImg32x32_best.pth", 3)
STEM_DOWN_MODEL = Single_Stem_Classifier("training/stemdownImg32x32_best.pth", 3)
MIN_BAR_HEIGHT = 8
ORIGINAL_IMAGE_RESIZE_RATIO = 2
REST_CLASSIFIER = Rest_Classifier(r'training\rest_remain2x1_best.pth', 5)
SFN_CLEF_CLASSIFIER = Sfn_Clef_classifier('training/best_model32x32_seg_dil4x4.pth', 7)


# --------------------------------------------------------------------------------------------------
# debug images
# --------------------------------------------------------------------------------------------------
def outputDebugStemRestImg(dataDict):
    img = np.ones_like(dataDict['image'])*255
    ys,xs = np.where(dataDict['stems_rests']>0)
    img[ys,xs] = (0,0,255)
    # outputImWrite(f'{img_name}_stemrests.jpg', img)
    imwrite(f'stemrests_.jpg', img)

def outputDebugSymbolsImg(dataDict):
    img = np.ones_like(dataDict['image'])*255
    ys,xs = np.where(dataDict['symbols']>0)
    img[ys,xs] = (255,0,0)
    # outputImWrite(f'{img_name}_symbols.jpg', img)
    imwrite(f'symbols_.jpg', img)

def outputNotegroupStemMapImg(noteGroupStemMap, image):
    ys,xs = np.where(noteGroupStemMap>0)
    zz = image.copy()
    zz[ys,xs,1:2] = 100
    # outputImWrite(f'{img_name}_noteGroupStemMap.jpg', zz)
    imwrite(f'noteGroupStemMap.jpg', zz)

def outputNoteGroupMapImg(noteGroupMap, image):
    ys,xs = np.where(noteGroupMap>0)
    zz = image.copy()
    zz[ys,xs,1:2] = 100
    # outputImWrite(f'{img_name}_noteGroupStemMap.jpg', zz)
    imwrite(f'noteGroupMap.jpg', zz)

def outputNoteMapImg(beamMapImg, image, staffObjList):
    colors = [(0,0,255),(0,255,0),(255,255,0), (0,120,230)]
    _,_,rr = cv2.split(beamMapImg)
    zz = image.copy()
    for i in range(len(staffObjList)):
        ys,xs = np.where(rr==(i+1))
        zz[ys,xs] = colors[i%4]
    imwrite('GroupMap.jpg',zz)

def outputSfnClefNoteWhiteImg(beamMapImg, image, bar_height, noteGroupMap, staffObjList):
    bb,gg,_ = cv2.split(beamMapImg)
    _,zz = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    zzmask = zz.copy()
    zzmask[:max(staffObjList[0].ys[0]-bar_height*2,0),:] = (255,255,255)
    zzmask[min(staffObjList[-1].ys[-1]+bar_height*2,zzmask.shape[0]):,:] = (255,255,255)
    for sf in staffObjList:
        xMaskOut = np.where(np.max(noteGroupMap[max(sf.ys[0]-2*bar_height, 0):min(sf.ys[-1]+2*bar_height, zzmask.shape[0]), :],0)>0)[0]
        zzmask[max(sf.ys[0]-2*bar_height, 0):min(sf.ys[-1]+2*bar_height,zzmask.shape[0]),xMaskOut] = (255,255,255)
    ys,xs = np.where(gg>0)
    zzmask[ys,xs] = (255,255,255)
    bb = cv2.dilate(bb, np.ones((bar_height//2, bar_height//2), dtype= np.uint8), iterations=1)
    ys,xs = np.where(bb>=255)
    zzmask[ys,xs] = (255,255,255)
    # zzmask = cv2.dilate(zzmask, np.ones((bar_height//3,1),dtype=np.uint8),iterations=1)
    # zzmask = cv2.erode(zzmask, np.ones((bar_height//3,bar_height//3),dtype=np.uint8),iterations=1)
    # cv2.imwrite('test.jpg',zzmask)
    imwrite('SfnClefNoteWhite.jpg', zzmask)

def outputSfnClefNoteImg(image, beamMapImg):
    _,gg,_ = cv2.split(beamMapImg)
    _,zz = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    for i in range(1,4):
        ys,xs = np.where(gg==i)
        zz[ys,xs,(i-2)%3] = 100
        zz[ys,xs,(i-3)%3] = 0
    ys,xs = np.where(image[:,:,0]<128)
    zz[ys,xs,:] = (0,0,0)
    # outputImWrite(f'{img_name}_SfnClefNote.jpg', zz)
    imwrite(f'SfnClefNote.jpg', zz)

def outputSfnClefNoteBarlineRestImg(image, beamMapImg):
    zz = image.copy()
    _,gg,_ = cv2.split(beamMapImg)
    colors = [(190,120,0),(255,50,0),(50,240,0),(0,120,230),(0,0,255)]
    for i in range(1,6):
        ys,xs = np.where(gg==i)
        zz[ys,xs] = colors[i%5]

    imwrite(f'SfnClefNoteBarlineRest.jpg', zz)

def outputThingsBeforeDots(image, beamMapImg):
    _,gg,_ = cv2.split(beamMapImg)
    colors = [(230,200,130),(230,130,100),(120,170,140),(235,176,113),(0,0,255)]
    zz = image.copy()
    for i in [1,2,3,5]:
        ys,xs = np.where(gg==i)
        zz[ys,xs] = colors[i%5]
    ys,xs = np.where(image[:,:,1]<128)
    zz[ys,xs] = (0,0,0)
    i = 4
    ys,xs = np.where(gg==i)
    zz[ys,xs] = colors[i%5]
    imwrite(f'DotPrevious.jpg', zz)
    return zz



# --------------------------------------------------------------------------------------------------
# functions
# --------------------------------------------------------------------------------------------------
def init_bar_height(dataDict, min_barheight):
    def staffs_to_omrStaffList(staffs):
        left = min([sf.x_left for sf in staffs[0,:]])
        right = max([sf.x_right for sf in staffs[-1,:]])
        omrstaff_list: List[Staff] = []
        staffRange = []
        mindiffs = []
        maxdiffs = []
        for i in range(staffs.shape[1]):
            yuppers = [sf.y_upper for sf in staffs[:,i]]
            ybottoms = [sf.y_lower for sf in staffs[:,i]]
            top = statistics.median(yuppers)
            bottom = statistics.median(ybottoms)
            unit_size = statistics.median([sf.unit_size for sf in staffs[:,i]])
            minMaxDiff = min(max(yuppers)-min(yuppers), max(ybottoms)-min(ybottoms))
            maxMaxDiff = max(max(yuppers)-min(yuppers), max(ybottoms)-min(ybottoms))
            ys = [int(top), int(top+unit_size), int((top+bottom)/2),int(bottom-unit_size),int(bottom)]
            sf = Staff(int(left), int(right), ys, minMaxDiff, yuppers, ybottoms)
            mindiffs.append(minMaxDiff)
            maxdiffs.append(maxMaxDiff)
            omrstaff_list.append(sf)
            staffRange.append(range(int(top),int(bottom)))
        return omrstaff_list, staffRange
    staff = dataDict['staff']
    staffs, zones = staff_extract_staffobj(staff, min_barheight)
    staffList, staffRange = staffs_to_omrStaffList(staffs)
    barheight = [sf.get_yOne_float() for sf in staffList]
    avg_barheight = int(sum(barheight)/len(barheight))
    return avg_barheight, staffList

def getBeamImage(dataDict, bar_height, staffObjList:List[Staff]):
    stems_rests = dataDict['stems_rests']
    clefs_keys = dataDict['clefs_keys']
    notehead = dataDict['notehead']
    symbols = dataDict['symbols']
    image:np.ndarray = dataDict['image']

    _, img128 = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((bar_height//3, bar_height//3),dtype=np.uint8)
    beam = cv2.erode(img128, kernel, iterations = 1)
    imwrite('note_beam1.jpg',beam)

    for ll in range(3):
        haslinecount = 0
        totalCount = 0
        for staff in staffObjList:
            for b in range(len(staff.ys)):
                curr = sum(np.max(beam[max(staff.ys[b]-bar_height//3,0):min(staff.ys[b]+bar_height//3, beam.shape[0]), staff.left:staff.right,1],0)==255)
                if curr>(staff.right-staff.left)//3:
                    haslinecount+=1
                totalCount+=1
        if haslinecount/totalCount>0.6:
            kernel = np.ones((bar_height//4, bar_height//4),dtype=np.uint8)
            beam = cv2.erode(beam, kernel, iterations = 1)
        else:
            break
    imwrite('note_beam1.5.jpg',beam)
    xs,ys = np.where(clefs_keys>0)
    beam[xs,ys] = (0,0,0)
    beamWithoutStemRest = beam.copy()
    imwrite('note_beam2.jpg',beamWithoutStemRest)
    # beamret is beamNoClefKey
    xs,ys = np.where(stems_rests>0)
    beam[xs,ys] = (255,255,255)
    imwrite('note_beam3.jpg', beam)
    # beam is beam with stem_rests

    kernel_tall = np.ones((1,bar_height*2), np.uint8)
    beam_nostem = cv2.erode(beam, kernel_tall)
    beam_nostem = cv2.dilate(beam_nostem, kernel_tall)
    imwrite('note_beam4.jpg', beam_nostem)
    beam_nostem_nostaff = beam_nostem.copy()
    savebeambw = cv2.cvtColor(beam_nostem, cv2.COLOR_BGR2GRAY)
    toDel = np.where(np.sum(savebeambw>127,axis=1)>savebeambw.shape[1]*0.4)[0] # VARIABLE threshold to delete
    for d in toDel:
        beam_nostem_nostaff[d,:] = 0
    imwrite('note_beam5.jpg', beam_nostem_nostaff)
    # beam nostem is beam with stemRest + erode (also has notes)
    return beam, beamWithoutStemRest, beam_nostem_nostaff

def generateBeamStaffImg(savebeam:np.ndarray, staffObjList:List[Staff], barheight:int,stepSize=1):
    # beamThick = cv2.dilate(savebeam, np.ones((3,1), dtype=np.uint8), iterations=1)    
    mappingImgRgb = np.zeros((savebeam.shape[0], savebeam.shape[1], 3))
    savebeambw = cv2.cvtColor(savebeam, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(savebeambw.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    selectedContours = []
    bb,gg,rr = cv2.split(mappingImgRgb)
    staffYCenters = []
    for idx,staff in enumerate(staffObjList):
        if staff.left <= 4:
            staff.left = 5
        if staff.right >= savebeam.shape[1]-3:
            staff.right = savebeam.shape[1]-4
        staffYs = staff.ys
        oneY:float = staff.get_yOne_float()
        staffYCenter = staffYs[2]
        rr = cv2.rectangle(rr, (staff.left, staffYs[0]), (staff.right, staffYs[-1]), idx+1,-1)
        # -1: how far to the top [0]
        rr[staffYs[0]:min(staffYs[-1]+barheight, savebeam.shape[0]),-1] = np.array(range(0, min(staffYs[-1]+barheight, savebeam.shape[0])-staffYs[0]))
        rr[max(staffYs[0]-barheight, 0):staffYs[-1],-2] = np.array(range(staffYs[-1]-max(staffYs[0]-barheight, 0),0,-1))
        # line one's index will be 1~10:1, 2s will be 20~200:20
        indexRatio = (idx%2)*20+(1-idx%2)*1
        staffYCenter = staffYs[2]
        staffYCenters.append(staffYCenter)

        for i in range(1,13):
            if staffYCenter-i*oneY>0:
                rr[round(staffYCenter-i*oneY):round(staffYCenter-(i-1)*oneY),(idx+1)%4] = i+12
            else:
                break
        for i in range(1,13):
            if staffYCenter+i*oneY<rr.shape[0]:
                rr[round(staffYCenter+(i-1)*oneY):round(staffYCenter+i*oneY),(idx+1)%4] = 13-i
            else:
                break
    staffSeperationYs = [(staffYCenters[i]+staffYCenters[i-1])//2 for i in range(1,len(staffYCenters))]
    staffSeperationYs.insert(0,0)
    staffSeperationYs.append(savebeam.shape[0])
    # now it's the threshold for each line of staff
    for staffIdx in range(1,len(staffSeperationYs)):
        rr[staffSeperationYs[staffIdx-1]:staffSeperationYs[staffIdx], 4] = staffIdx
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w>barheight*1.5 and w<savebeam.shape[1]//3:
            area = cv2.contourArea(cnt)
            if area>w and not (w<2*barheight and h/w>0.5)and w>h:
                selectedContours.append(cnt)
    bb = cv2.drawContours(bb, selectedContours,-1, 255,thickness=-1)
    imwrite('beamOri.jpg',savebeambw)
    mappingImgRgb = cv2.merge([bb,gg,rr])
    imwrite('beamOricontours.jpg',mappingImgRgb)
    bbUint8 = bb.astype(np.uint8)
    kernel = np.ones((1,barheight//3), dtype=np.uint8)
    bbUint8 = cv2.dilate(bbUint8, kernel, iterations=1)
    mappingImgUint8= cv2.merge([bbUint8,gg.astype(np.uint8),rr.astype(np.uint8)])
    print('preparing for beam image')
    for xIdx in range(0,savebeam.shape[1], stepSize*2):
        # no white in this place
        if sum(mappingImgUint8[:,xIdx,0]) == 0:
            continue
        startingY = np.where(mappingImgUint8[:,xIdx,0]==255)[0].tolist()
        # the ending of the bottommost one
        startingY.append(min(mappingImgUint8.shape[0],startingY[-1]+barheight*5))
        for i in range(0,len(startingY)-1):
            yStartVal = 253
            for yIdx in range((startingY[i]+stepSize)//(2*stepSize)*(2*stepSize)+stepSize, startingY[i+1],stepSize*2):
                if yStartVal<=0:
                    break
                mappingImgUint8[yIdx,xIdx,0]=yStartVal
                yStartVal-=stepSize*2
        # the ending of the topmost one
        startingY.pop()
        startingY.insert(0, max(startingY[0]-barheight*5, 0))
        for i in range(len(startingY)-1, 0,-1):
            yStartVal = 254
            for yIdx in range((startingY[i]-1)//(2*stepSize)*(2*stepSize), startingY[i-1],-stepSize*2):
                if yStartVal<=0:
                    break
                mappingImgUint8[yIdx,xIdx,0]=yStartVal
                yStartVal-=stepSize*2
    onlyBImg = mappingImgUint8.copy()
    onlyBImg[:,:,1] = onlyBImg[:,:,0]
    onlyBImg[:,:,2] = onlyBImg[:,:,0]
    imwrite('onlyBImg.jpg',onlyBImg)
    saveImages = {}
    saveImages['gradImg']=onlyBImg
    saveImages['beamContour']=mappingImgRgb
    return mappingImgUint8, saveImages

def getInitialNoteheadBoxList(dataDict, beam_nostem, beamMapImg:np.ndarray, bar_height):
    
    def get_bbox(data: np.ndarray) -> List[Tuple[int,int,int,int]]:
        contours, _ = cv2.findContours(data.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            box = (x, y, x+w, y+h)
            bboxes.append(box)
        return bboxes

    def filter_small_bbox_area(data: np.ndarray, noteimg:np.ndarray, xmin, ymin, areamin):
        filtered_Box = []
        for box in data:
            x0,y0,x1,y1 = box
            w = x1-x0
            h = y1-y0
            if w>=xmin and h>=ymin and w*h>areamin:
                filtered_Box.append(box)
            else:
                noteimg[y0:y1,x0:x1] = 0
        return filtered_Box,noteimg

    def split_connected_notes(bboxList, bar_height,image):
        minh = int(bar_height*0.9)
        minw = int(bar_height)
        retlst = []
        for bbox in bboxList:
            x0,y0,x1,y1 = bbox
            wrat = (x1-x0)/minw
            hrat = (y1-y0)/minh
            wnum = int(wrat)
            hnum = int(hrat)
            note_w = int(minw*1.3)
            note_h = int(minh*1.3)
            if wnum*hnum <= 1:
                retlst.append(bbox)
            elif wnum>=2: 
                if y1-y0<=bar_height or wnum>=3:
                    xwid = (x1-x0)//wnum
                    for i in range(wnum):
                        retlst.append((x0+i*xwid,y0,x0+(i+1)*xwid,y1))
                # it's a scatter of three vertically stacked notes with two adjacent notes
                elif hnum>2.5 and hnum>wnum:
                    # first get the top and bottom ones
                    # if the top part has more black on the left (x0)
                    option2s = []
                    if np.sum(image[y0:y0+note_h,x0:x0+note_w])>np.sum(image[y0:y0+note_h,x1-note_w:x1]):
                        retlst.append((x0,y0,x0+note_w, y0+note_h))
                        # the other side go down once
                        option2s.append((x1-note_w,y0+note_h//2,x1, y0+int(note_h*1.5)))
                    else:
                        retlst.append((x1-note_w,y0,x1, y0+note_h))
                        option2s.append((x0,y0+note_h//2,x0+note_w, y0+int(note_h*1.5)))
                    # if the bottom part has more black on the left (x0)
                    if np.sum(image[y1-note_h:y1,x0:x0+note_w])>np.sum(image[y1-note_h:y1,x1-note_w:x1]):
                        retlst.append((x0,y1-note_h,x0+note_w, y1))
                        option2s.append((x1-note_w,y1-int(note_h*1.5),x1, y1-note_h//2))
                    else:
                        retlst.append((x1-note_w,y1-note_h,x1, y1))
                        option2s.append((x0,y1-int(note_h*1.5),x0+note_w, y1-note_h//2))
                    # deal with the middle one    
                    if np.sum(image[option2s[0][1]:option2s[0][3], option2s[0][0]:option2s[0][2]]) > np.sum(image[option2s[1][1]:option2s[1][3], option2s[1][0]:option2s[1][2]]):
                        retlst.append(option2s[0])
                    else:
                        retlst.append(option2s[1])
                else:
                    # if the left part has more black on top(y0)
                    if np.sum(image[y0:y0+note_h,x0:x0+note_w])>np.sum(image[y1-note_h:y1,x0:x0+note_w]):
                        retlst.append((x0,y0,x0+note_w, y0+note_h))
                    else:
                        retlst.append((x0,y1-note_h,x0+note_w, y1))
                    # if the right part has more black on top(y1)
                    if np.sum(image[y0:y0+note_h,x1-note_w:x1])>np.sum(image[y1-note_h:y1,x1-note_w:x1]):
                        retlst.append((x1-note_w,y0,x1, y0+note_h))
                    else:
                        retlst.append((x1-note_w,y1-note_h,x1, y1))

            elif hnum>=2:
                ywid = (y1-y0)//hnum
                for i in range(hnum):
                    retlst.append((x0,y0+i*ywid,x1,y0+(i+1)*ywid))
        return retlst

    def filter_boxes_on_beam(notehead_boxes, beamMapImg, areamin):
        filtered_Boxes = []
        for box in notehead_boxes:
            x0,y0,x1,y1 = box
            if np.max(beamMapImg[y0:y1, x0:x1, 0])==255:
                w = x1-x0
                h = y1-y0
                if w*h<areamin:
                    continue
            filtered_Boxes.append(box)
        return filtered_Boxes
    notehead = dataDict['notehead']
    image:np.ndarray = dataDict['image']
    xs,ys = np.where(cv2.cvtColor(beam_nostem, cv2.COLOR_BGR2GRAY)>0)
    notehead_mod = notehead.copy()*255
    notehead_mod[xs,ys] = 0

    imwrite('notehead_ori.jpg', notehead_mod)
    dilation_ratio1 = 0.33
    dilation_ratio2 = 0.4
    # dilate->erode twice to clear image + get clear noteheads
    size1 = int(round(bar_height*dilation_ratio1))
    note_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size1, size1))
    notehead_mod = cv2.erode(cv2.dilate(notehead_mod.astype(np.uint8), note_kernel), note_kernel)
    size2 = int(round(bar_height*dilation_ratio2))

    note_kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size2,size2))
    notehead_mod2 = cv2.erode(notehead_mod, note_kernel2)
    note_kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size2+1, size2+1))
    notehead_mod2 = cv2.dilate(notehead_mod2, note_kernel3)

    # both b&w image, white is where there's stuff
    imwrite('notehead_mod.jpg',notehead_mod)
    imwrite('notehead_mod2.jpg',notehead_mod2)
    # notehead mod is basically the cleaned beams

    # get the boxes, if the boxes is in 
    notehead_boxes_all = get_bbox(notehead_mod2)
    notehead_boxes_big, notehead_mod3= filter_small_bbox_area(notehead_boxes_all, notehead_mod2,xmin = bar_height*0.5,ymin = bar_height*0.2, areamin = bar_height*bar_height*0.4)
    imwrite('notehead_mod3.jpg',notehead_mod3)
    notehead_boxes = split_connected_notes(notehead_boxes_big, bar_height,notehead_mod2)
    notehead_boxes_filtered = filter_boxes_on_beam(notehead_boxes, beamMapImg,areamin = bar_height*bar_height*0.8)
    return notehead_boxes_filtered, notehead_mod2

def maskImage(noteheadBoxesInitList:List[Tuple[int,int,int,int]], noteBwImageOrigin: np.ndarray):
    maskedNoteBw = np.zeros_like(noteBwImageOrigin)
    for box in noteheadBoxesInitList:
        x0,y0,x1,y1 = box
        maskedNoteBw[y0:y1, x0:x1] = noteBwImageOrigin[y0:y1, x0:x1]
    imwrite('maskedNoteBw.jpg',maskedNoteBw)
    return maskedNoteBw

# noteheadInitial : b&w 2d image, 255: notehead, 
# beamMapImg: gradient image, the 1st dimension is what we care about
def getStemList(dataDict, bar_height, notehead_boxes:List[Tuple[int,int,int,int]], beamWithStemRest):
    image:np.ndarray = dataDict['image']

    drawimg = image.copy()
    # 255 is where there's stuff, 0 is empty
    _, img200 = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY_INV)
    imgh = image.shape[0]
    notehead_rectangle = np.zeros((image.shape[0],image.shape[1]))
    stem_init_lst:List[Stem]=[]
    # beambw is suppose to be white where there's beam
    beambw = cv2.cvtColor(beamWithStemRest, cv2.COLOR_BGR2GRAY)
    imwrite('beambw.jpg',beambw)
    # get the stem's directions
    for idx,boxOriginal in enumerate(notehead_boxes):
        x0,y0,x1,y1 = boxOriginal
        xleft = (x0+x1*2)//3
        xright = (x0*2+x1)//3

        leftRange = np.where(img200[y0:y1,xleft]==255)[0].tolist()
        rightRange = np.where(img200[y0:y1, xright] ==255)[0].tolist()
        if len(leftRange)*len(rightRange) != 0:
            y1 = y0+max(max(leftRange),max(rightRange))
            y0 += min(min(leftRange), min(rightRange))
            if y1<=y0:
                x0,y0,x1,y1 = boxOriginal
        box = (x0,y0,x1,y1)
        notehead_rectangle[y0:y1,x0:x1] = 1
        length = bar_height*3
        width = (x1-x0)//3
        leftXrange = range(x0-width//2,x0+width+1)
        rightXrange = range(x1-width, x1+width//2+1)
        topYrange = range(max(y0-length, 0),y0)
        bottomYrange = range(y1,min(y1+length, imgh))
        left_area = beambw[bottomYrange.start:bottomYrange.stop,
                         leftXrange.start: leftXrange.stop]    
        left_centerX = np.argmax(np.sum(left_area,0))
        right_area = beambw[topYrange.start:topYrange.stop,
                         rightXrange.start:rightXrange.stop]
        right_centerX = np.argmax(np.sum(right_area,0))
        # has to be more strict not too far to the left
        leftup_area = beambw[topYrange.start:topYrange.stop,
                         leftXrange.start: leftXrange.stop]
        leftup_centerX = np.argmax(np.sum(leftup_area,0))
        rightdn_area = beambw[bottomYrange.start:bottomYrange.stop,
                              rightXrange.start:rightXrange.stop]
        rightdn_centerX = np.argmax(np.sum(rightdn_area,0))
        isup = True
        hasStem = True
        alterBox = []
        leftRightRatio = np.sum(left_area[:,left_centerX])/np.sum(right_area[:, right_centerX])
        if leftRightRatio>1:
            # stem is in left bottom
            maxRange = [leftXrange, bottomYrange]
            isup = False
            if leftRightRatio<1.25:
                alterBox.append([rightXrange.start+right_centerX,topYrange.start,rightXrange.start+right_centerX,topYrange.stop])
        else:
            # it's either close enough or in right bottom
            maxRange = [rightXrange, topYrange]
            # they are close enough -> alterbox will always record the left bottom one
            if leftRightRatio>0.8:
                alterBox.append([leftXrange.start+left_centerX,bottomYrange.start,leftXrange.start+left_centerX,bottomYrange.stop])
        maxArea = max(np.sum(left_area[:,left_centerX]),np.sum(right_area[:, right_centerX]))
        maxAlterArea = max(np.sum(leftup_area[:,leftup_centerX]),np.sum(rightdn_area[:,rightdn_centerX]))
        if maxAlterArea>255*length//2 and maxAlterArea/maxArea>1.5:
            if np.sum(leftup_area[:,leftup_centerX])>np.sum(rightdn_area[:,rightdn_centerX]):
                alterBox.append([leftXrange.start+leftup_centerX, topYrange.start, leftXrange.start+leftup_centerX, topYrange.stop])
            else:
                alterBox.append([rightXrange.start+rightdn_centerX, bottomYrange.start, rightXrange.start+rightdn_centerX, bottomYrange.stop])
        if len(alterBox)==0 and maxArea < 255*length//3:
            hasStem = False
            # hasStem is a flag for later grouping notes
        xrange, yrange = maxRange
        crop = img200[yrange.start:yrange.stop, xrange.start:xrange.stop]
        a = (np.where(np.sum(crop,0)==np.max(np.sum(crop,0)))[0]).tolist()
        chosenx = a[len(a)//2]
        x_y = (xrange.start+chosenx, y0 if isup else y1)
        stem_init_lst.append(Stem(x_y, isup,boxOriginal,box, hasStem=hasStem, alterbox=alterBox))
        if len(alterBox)==0 and hasStem:
            drawimg = cv2.rectangle(drawimg, (xrange.start+chosenx-1, yrange.start),(xrange.start+chosenx+1,yrange.stop),(255,0,120), 1, cv2.LINE_AA)
        elif hasStem:
            drawimg = cv2.putText(drawimg, str(idx), (x0,y0),cv2.FONT_HERSHEY_SIMPLEX,  1, (120,0,120), 2, cv2.LINE_AA)
            drawimg = cv2.rectangle(drawimg, (xrange.start+chosenx-1, yrange.start),(xrange.start+chosenx+1,yrange.stop),(255,255,0), 2, cv2.LINE_AA)
            for ab in alterBox:
                drawimg = cv2.rectangle(drawimg, (ab[0], ab[1]),(ab[2],ab[3]),(0,255,0), 3, cv2.LINE_AA)
        drawimg = cv2.rectangle(drawimg, (x0,y0),(x1,y1),(0,0,255), 2, cv2.LINE_AA)
    imwrite('notehead_drawimg.jpg',drawimg)
    return stem_init_lst, image, {'notehead_drawimg':drawimg}

def assignStemLength(stem_init_lst:List[Stem], image, beamMapImg:np.ndarray, stepSize: int, bar_height):
    # get the stem's length
    img200 =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,img200bw = cv2.threshold(img200, 200, 255, cv2.THRESH_BINARY_INV)
    # img200 is black and white, white is where there's stem
    imwrite('img200inv.jpg', img200bw)
    # wid = 1
    topthres = bar_height//2
    ending_black_height = 3
    # make it whiter for better visualization
    drawimg2 = np.ones_like(img200bw)*255 - img200bw.copy()//4
    drawimg2 = cv2.cvtColor(drawimg2, cv2.COLOR_GRAY2BGR)
    bigStep = stepSize*2
    for idx,coordStart in enumerate(stem_init_lst):
        x,yOri = coordStart.start
        wid = (coordStart.noteBox[2]-coordStart.noteBox[0])//3
        yend = yOri
        sign = 1 if coordStart.isup else -1
        # is in staff +- 3.5 barheight, no alternative box, has stem
        # +beamMapImg[max(yOri-int(2.5*bar_height)*sign,0),x,2]
        isTypical:bool = (len(coordStart.alternativeBox) == 0) and coordStart.hasStem and min([abs(posIdx-11.5) for posIdx in beamMapImg[yOri,0:4,2]])<5

        # if the beamMap is less than 4*barheight away and the beam itself is somewhere in a staffArea, assign it
        # elif it's not too far away (2*4*barheight), see if the area has at least 1/2 blacks
        # else it's gonna go through the clssification model to see if it's a single quarter/eighth/sixteenth
        xLeft = x//bigStep*bigStep
        xRight = min(beamMapImg.shape[1]-1,x//bigStep*bigStep+bigStep)
        staffLineNumber:int = beamMapImg[yOri, 4, 2]
        barPositionI: int = beamMapImg[yOri, staffLineNumber%4, 2]
        stemUpLongest:int = 0 if (barPositionI<8 or barPositionI>17) else 17-barPositionI
        stemDownLongest:int = 0 if (barPositionI<8 or barPositionI>17) else barPositionI-8
        if coordStart.isup:
            longestNormal = stemUpLongest
            # go how far is considered normal and ok
            y = yOri//bigStep*bigStep-stepSize
            leftUpValue = beamMapImg[y,xLeft, 0]
            rightUpValue = beamMapImg[y,xRight, 0]
            if leftUpValue>rightUpValue:
                maxMapValue = leftUpValue
            else:
                maxMapValue = rightUpValue
            assert maxMapValue%2==1 or maxMapValue == 0
            if maxMapValue != 0:
                if isTypical and (253-maxMapValue)+bigStep<=(longestNormal)*bar_height:
                    if maxMapValue < 255:
                        yend = y-(253-maxMapValue)
                        assert max(beamMapImg[yend, xLeft,0],beamMapImg[yend, xRight,0]) >= 253
                        while max(beamMapImg[yend, xLeft,0],beamMapImg[yend, xRight,0]) >= 253-bar_height//2:
                            yend -= bigStep
                        while max(beamMapImg[yend, xLeft,0],beamMapImg[yend, xRight,0]) == 255:
                            yend -= 1
                        # the end of the stem doesn't have enough "stem (white)" (the image is inversed)
                        if (253-maxMapValue)+bigStep>(longestNormal-3)*bar_height and (sum(np.max(img200bw[yend:y, x-wid:x+wid+1],1))<(y-yend)*255*0.8 or sum(np.max(img200bw[yend:(y+yend)//2, x-wid:x+wid+1],1))<(y-yend)*255*0.4):
                            yend = yOri
            if yend==yOri:
                if isTypical and coordStart.hasStem and len(coordStart.alternativeBox)==0:
                    yend -= min(3,longestNormal)*bar_height
                # while there is still things in the inversed bw original image
                while sum(np.max(img200bw[yend-ending_black_height:yend, x-wid:x+wid+1],1))!=0:
                    yend -= 1
            else:
                while sum(np.max(img200bw[yend-ending_black_height:yend, x-1:x+2],1))!=0: # and max([(10-beamMapImg[yend,x,1]%11)%10,(10-beamMapImg[yend,x,1]//20)%10]): 
                    yend -= 1
        else:
            longestNormal = stemDownLongest
            y = min(yOri//bigStep*bigStep+bigStep,beamMapImg.shape[0]-1)
            leftUpValue = beamMapImg[y,xLeft, 0]
            rightUpValue = beamMapImg[y,xRight, 0]
            if leftUpValue>rightUpValue:
                maxMapValue = leftUpValue
            else:
                maxMapValue = rightUpValue
                xRight = x//bigStep*bigStep+bigStep
            assert maxMapValue%2==0 or maxMapValue == 255
            if isTypical and (254-maxMapValue)+bigStep<=longestNormal*bar_height:
                if maxMapValue < 255:
                    yend = y+(254-maxMapValue)
                    assert max(beamMapImg[yend, xLeft,0],beamMapImg[yend, xRight,0]) >=254
                    while max(beamMapImg[yend, xLeft,0],beamMapImg[yend, xRight,0]) >= 254-bar_height//2 and (yend-yOri)<=(longestNormal)*bar_height:
                        yend += bigStep
                    while max(beamMapImg[yend, xLeft,0],beamMapImg[yend, xRight,0]) == 255 and (yend-yOri)<=(longestNormal)*bar_height:
                        yend += 1
                    # the end of the stem doesn't have enough "stem (white)" (the image is inversed)
                    if (254-maxMapValue)+bigStep>(longestNormal-3)*bar_height and (sum(np.max(img200bw[y:yend, x-wid:x+wid+1],1))<(yend-y)*128 or sum(np.max(img200bw[(y+yend)//2:yend, x-wid:x+wid+1],1))<(yend-y)*64):
                        yend = yOri
            if yend==yOri:
                if isTypical and coordStart.hasStem and len(coordStart.alternativeBox)==0:
                    yend += min(3,longestNormal)*bar_height
                # while there is still things in the inversed bw original image
                while sum(np.max(img200bw[yend:yend+ending_black_height, x-wid:x+wid+1],1))!=0 and (yend-yOri)<=(longestNormal)*bar_height: 
                    yend += 1
            else:
                while sum(np.max(img200bw[yend:yend+ending_black_height, x-1:x+2],1))!=0: # and min([beamMapImg[yend,x,1]%11,beamMapImg[yend,x,1]//20])<2:
                    yend += 1
        if abs(yend-y)<bar_height and not coordStart.hasStem:
            yend = yOri
        coordStart.setYlen(yend-yOri)
        assert coordStart.end is not None
        drawimg2 = cv2.rectangle(drawimg2, coordStart.getTopCoord(), coordStart.getBottomCoord(),(255,0,255), 1, cv2.LINE_AA)
        if not isTypical:
            drawimg2 = cv2.putText(drawimg2, str(idx), coordStart.end, cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1, cv2.LINE_AA)
        else:
            drawimg2 = cv2.putText(drawimg2, str(idx), coordStart.end, cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1, cv2.LINE_AA)
    imwrite('notestem_drawimg.jpg',drawimg2)
    return stem_init_lst, {'notestem_drawimg': drawimg2}

def removeNotes(maskedNoteBwImage:np.ndarray, beamNoClefKeyWithStemRest:np.ndarray, beamMapImg:np.ndarray):
    beambw = beamMapImg[:,:,0]
    xs,ys = np.where(maskedNoteBwImage>0) # where there's note but no beam
    beamNoNotes = beamNoClefKeyWithStemRest.copy()
    beamNoNotes[xs,ys] = (0,0,0)
    imwrite('beamNoNotes1.jpg',beamNoNotes)
    xs,ys = np.where(beambw==255)
    beamNoNotes[xs,ys] = (255,255,255)
    imwrite('beamNoNotes2.jpg',beamNoNotes)

    return beamNoNotes


def assignBeamLengthBeamImg(stem_list:List[Stem], image:np.ndarray, beam:np.ndarray, barheight:int):
    x_diff = barheight//2
    beam = cv2.cvtColor(beam, cv2.COLOR_BGR2GRAY)
    # if it's completely white in the white_buffer_max area then stop
    white_buffer_max = barheight
    beam_start_buffer_max = barheight//3
    for stem in stem_list:
        xCenter,yCenter = stem.start
        xLeft = xCenter-x_diff
        xRight = min(image.shape[1]-1,xCenter+x_diff)
        yMax = stem.getYLen()
        if stem.getYLen() == 0:
            continue
        if stem.isup:
            for xtype in ['left', 'right']:
                if xtype=='left':
                    currX = xLeft
                else:
                    currX = xRight
                currY = stem.getTopCoord()[1]
                # if the starting point of beam is not white, meaning that it might be a symbol, staff line etc
                # keep going down until it's at the right place
                if sum(np.max(beam[currY:currY+beam_start_buffer_max, xCenter-barheight//2:xCenter+barheight//2],1)) < 255*beam_start_buffer_max:
                    while(sum(np.max(beam[currY:currY+beam_start_buffer_max, xCenter-barheight//2:xCenter+barheight//2],1)) < 255*beam_start_buffer_max and currY<stem.getBottomCoord()[1]-barheight//2):
                        currY = currY+1
                stem.end = (stem.end[0], currY)

                if beam[currY, currX] == 0:
                    while(beam[currY, currX] == 0 and currY>stem.getBottomCoord()[1]+barheight//2):
                        currY = currY+1
                else:
                    while beam[currY-1, currX] == 255:
                        currY = currY-1
                # now the topStartY is the "real start"
                stem.setBeam(typ=xtype, top=currY)
                # go down until it's wholely black
                while sum(beam[currY:currY+white_buffer_max, currX])>255:
                    currY+=1
                stem.setBeam(typ=xtype, bottom=currY)
        else:
            for xtype in ['left', 'right']:
                if xtype=='left':
                    currX = xLeft
                else:
                    currX = xRight
                currY = stem.getBottomCoord()[1]

                # if the starting point of beam is not white, meaning that it might be a symbol, staff line etc
                # keep going up until it's at the right place
                if sum(np.max(beam[currY-beam_start_buffer_max:currY, xCenter-barheight//2:xCenter+barheight//2],1)) < 255*beam_start_buffer_max:
                    while(sum(np.max(beam[currY-beam_start_buffer_max:currY, xCenter-barheight//2:xCenter+barheight//2],1)) < 255*beam_start_buffer_max and currY>stem.getTopCoord()[1]+barheight//2):
                        currY = currY-1
                stem.end = (stem.end[0], currY)

                # if it's black (need to go up)
                if beam[currY, currX] == 0:
                    while(beam[currY, currX] == 0 and currY<stem.getTopCoord()[1]-barheight//2):
                        currY = currY-1
                else:
                    while beam[currY+1, currX] == 255:
                        currY = currY+1
                # now the currY is the "real start"
                stem.setBeam(typ=xtype, bottom=currY)
                # go up until it's wholely black
                while sum(beam[currY-white_buffer_max:currY, currX])>255:
                    currY-=1
                stem.setBeam(typ=xtype, top=currY)
    imgBgr = image.copy()
    beam_heights = []
    for stem in stem_list:
        beam_heights.append(stem.leftBeam[1]-stem.leftBeam[0])
        beam_heights.append(stem.rightBeam[1]-stem.rightBeam[0])
        imgBgr = cv2.rectangle(imgBgr, (stem.start[0]-x_diff, stem.leftBeam[0]), (stem.start[0]-x_diff, stem.leftBeam[1]),(0,0,255),3,cv2.LINE_AA)
        imgBgr = cv2.rectangle(imgBgr, (stem.start[0]+x_diff, stem.rightBeam[0]), (stem.start[0]+x_diff, stem.rightBeam[1]),(255,255,0),3, cv2.LINE_AA)
    imwrite('leftrightRectBeam.jpg', imgBgr)
    return imgBgr, stem_list, beam_heights


def knnRhythmAndDraw(stem_list_assigned:List[Stem], beam_heights, image, bar_height, beamMapImg, stemUpClassifier, stemDownClassifier,img_name):
    def get_beam_height(beam_heights:List[int], barheight):
        whole_thres = [1, int(barheight*0.6), int(barheight*1.8), int(barheight*2.7), int(barheight*3.2)]
        if max(beam_heights) == 0:
            return beam_heights
        count, _ = np.histogram(beam_heights, max(beam_heights))
        # expand the list to be able to access list[init_thres[-1]]
        init_thres = whole_thres[:sum([0 if thres>len(count) else 1 for thres in whole_thres])+1]
        expanded_count = list(count) + [0]*(init_thres[-1]+1-len(count))
        while len(init_thres)>0 and sum(expanded_count[init_thres[-1]:])<max(len(beam_heights)//100,5):
            tooBig = init_thres.pop(-1)
            beam_heights = [tooBig if bh>tooBig else bh for bh in beam_heights]
            expanded_count[tooBig:] = [0]*len(expanded_count[tooBig:])
        init_thres.append(len(expanded_count))
        centers = [0]*len(init_thres)
        for i in range(1,len(init_thres)):
            centers[i] = np.argmax(expanded_count[init_thres[i-1]:init_thres[i]])+init_thres[i-1]
        kmeans = KMeans(n_clusters = len(centers), init = np.array(centers, dtype=np.uint8).reshape(-1, 1), n_init = 1)
        kmeans.fit(np.array(beam_heights,dtype=np.uint8).reshape(-1,1))
        return kmeans.labels_.tolist()
    def getSingleStemPred(img, model:Single_Stem_Classifier):
        # ['n2', 'n4', 'n816'] (0,1,2)if it's flat then it should be n1 3
        if img.shape[1]/img.shape[0]>1.5:
            return 3
        else:
            return model.predict(img)

    beam_type = get_beam_height(beam_heights, bar_height)
    class_colors = [(255,0,0),(0,0,255),(255,255,0),(0,120,255),(40,255,40), (245, 220, 255),(230,130,175),(165,170, 70)]
    class_names = ['1/4','1/8','1/16','1/32', '1/64', '1/128', '1/2', '1/1']
    imgrgb = image.copy()
    x_diff = bar_height//2
    mapBeamVal, middle, mapStaffNum = cv2.split(beamMapImg)
    # the first one is a empty one just for easier indexing
    noteGroupStemMap = np.zeros((beamMapImg.shape[0], beamMapImg.shape[1]), dtype=np.uint16)
    noteGroupList:List[NoteGroup|None] = [None]
    img200 =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,img200bw = cv2.threshold(img200, 200, 255, cv2.THRESH_BINARY_INV)
    for j in range(0,len(class_names)):
        imgrgb = cv2.putText(imgrgb,class_names[j], (30, 30+j*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_colors[j], 2, cv2.LINE_AA)
    # stillLabel = True
    # init_label_folder()
    predLabelConvert = [-2,0,1,-1]
    for idx,stem in enumerate(stem_list_assigned):
        stem.rhythm = max(beam_type[idx*2: idx*2+2])
        leftHeight, rightHeight = stem.getBeamHeights()
        # solve the issue of misclassify something to no stem when there's one (since knn also train on 0s)
        if stem.rhythm==0 and max(leftHeight, rightHeight)>0:
            stem.rhythm = 1
        elif stem.rhythm == 0:
            sy0, sy1 = stem.getY0Y1()
            nx0, ny0, nx1, ny1 = stem.noteBox
            if stem.isup:
                x0 = nx0
                x1 = min(nx1+(nx1-nx0), img200bw.shape[1]-1)
                y0 = sy0
                y1 = ny1
            else:
                x0 = max(nx0-(nx1-nx0)//2,0)
                x1 = nx1
                y0 = ny0
                y1 = sy1
            img200Crop = img200bw[y0:y1, x0:x1]
            # ['n2', 'n4', 'n816'] (0,1,2), 3 if the box doesn't have stem
            predLabel = getSingleStemPred(img200Crop, model=stemUpClassifier if stem.isup else stemDownClassifier)
            stem.rhythm = predLabelConvert[predLabel]
            stem.isSingle = True
        xcenter = stem.start[0]
        if len(stem.alternativeBox)==0:
            stemX0 = xcenter - bar_height//3 #(bar_height*2//3 if stem.isup else bar_height//3)
            stemX1 = xcenter + bar_height//3 #(bar_height//3 if stem.isup else bar_height*2//3)
            stemY0,stemY1 = stem.getY0Y1()
        else:
            # x0,y0,x1,y1, x0 == x1
            x00, stemY0, xcenter2, stemY1 =stem.alternativeBox[0]
            assert x00 == xcenter2
            stemX0 = xcenter2 - bar_height//3
            stemX1 = xcenter2 + bar_height//3
        
        if stemY0 == stemY1:
            stemY1 = stemY0+1
        elif len(stem.alternativeBox)==0:
            # y0 is the top, smaller one
            stemY0 = stemY0 - beamMapImg[stemY0, -1, 2]
            stemY1 = stemY1 + beamMapImg[stemY1, -2, 2]
        cropPart = noteGroupStemMap[stemY0:stemY1, stemX0:stemX1]
        staffNumPart = mapStaffNum[stemY0:stemY1, stemX0:stemX1]
        staffNum = np.max(staffNumPart)
        # there's no overlapping group
        if np.max(cropPart) == 0:
            noteGroupStemMap[stemY0:stemY1, stemX0:stemX1] = len(noteGroupList)
            mapStaffNum[stemY0:stemY1, stemX0:stemX1] = staffNum
            noteGroupList.append(NoteGroup(stem))
        else:
            overlappingIndex = np.unique(cropPart).tolist()
            if 0 in overlappingIndex:
                overlappingIndex.remove(0)
            if len(overlappingIndex)==1:
                currGroupIdx = overlappingIndex[0]
                # xcenter-bar_height//2:xcenter+bar_height//2
                noteGroupStemMap[stemY0:stemY1, stemX0:stemX1] = currGroupIdx
                yy,xx = np.where(noteGroupStemMap == currGroupIdx)
                mapStaffNum[yy,xx] = np.max(mapStaffNum[yy,xx])
                noteGroupList[currGroupIdx] = mergeNoteGroup(noteGroupList[currGroupIdx], NoteGroup(stem), beamMapImg)
                # noteGroupList[currGroupIdx].addNoteStem(stem)
            else:
                currGroupIdx = overlappingIndex[0]
                noteGroupStemMap[stemY0:stemY1, stemX0:stemX1] = currGroupIdx
                for gIdx in overlappingIndex[1:]:
                    ys,xs = np.where(cropPart == gIdx)
                    noteGroupStemMap[ys,xs] = currGroupIdx
                    noteGroupList[currGroupIdx] = mergeNoteGroup(noteGroupList[currGroupIdx],noteGroupList[gIdx], beamMapImg)
                    noteGroupList[gIdx] = None
                yy,xx = np.where(noteGroupStemMap == currGroupIdx)
                mapStaffNum[yy,xx] = np.max(mapStaffNum[yy,xx])
                noteGroupList[currGroupIdx] = mergeNoteGroup(noteGroupList[currGroupIdx], NoteGroup(stem), beamMapImg)
    
    noteSizes = []
    for ng in noteGroupList:
        if ng is None:
            continue
        for stem in ng.noteStemList:
            imgrgb = cv2.rectangle(imgrgb, (stem.getX()-bar_height//3, stem.getTopCoord()[1]), (stem.getX()+bar_height//3, stem.getBottomCoord()[1]),
                                class_colors[stem.rhythm], 1, cv2.LINE_AA)
            x0,y0,x1,y1 = stem.noteBox
            noteSizes.append((x1-x0)*(y1-y0))
            imgrgb = cv2.rectangle(imgrgb, (x0,y0),(x1,y1), class_colors[stem.rhythm], 1, cv2.LINE_AA)
            if len(stem.alternativeBox) >0:
                alterBoxes = stem.alternativeBox
                for alterBox in alterBoxes:
                    imgrgb = cv2.rectangle(imgrgb, (alterBox[0], alterBox[1]),(alterBox[2],alterBox[3]),(120,150,255), 3, cv2.LINE_AA)
    kmeans = KMeans(n_clusters = 2, init = np.array([min(noteSizes), max(noteSizes)], dtype=np.uint16).reshape(-1, 1), n_init = 1)
    kmeans.fit(np.array(noteSizes,dtype=np.uint16).reshape(-1,1))
    
    if kmeans.cluster_centers_[1]/kmeans.cluster_centers_[0]>2:
        imgrgb2 = image.copy()
        currentCount = -1
        minSize = int(kmeans.cluster_centers_[1]//2)
        for ngId, ng in enumerate(noteGroupList):
            if ng is None:
                continue
            stmToRemove = []
            for stemId, stem in enumerate(ng.noteStemList):
                x0,y0,x1,y1 = stem.noteBox
                currentCount = currentCount+1
                # if kmeans.labels_[currentCount] == 0:
                #     break
                if (x1-x0)*(y1-y0)<minSize:
                    # TODO potentially will filter out the big notes that are grouped to small ones
                    for stem in ng.noteStemList:
                        stem.isOrnament = True
                    stmToRemove.append(stemId)
                    continue
                imgrgb2 = cv2.rectangle(imgrgb2, (stem.getX()-bar_height//3, stem.getTopCoord()[1]), (stem.getX()+bar_height//3, stem.getBottomCoord()[1]),
                                    class_colors[stem.rhythm], 1, cv2.LINE_AA)
                
                imgrgb2 = cv2.rectangle(imgrgb2, (x0,y0),(x1,y1), class_colors[stem.rhythm], 1, cv2.LINE_AA)
                if len(stem.alternativeBox) >0:
                    alterBoxes = stem.alternativeBox
                    for alterBox in alterBoxes:
                        imgrgb2 = cv2.rectangle(imgrgb2, (alterBox[0], alterBox[1]),(alterBox[2],alterBox[3]),(120,150,255), 3, cv2.LINE_AA)
            stmToRemove.reverse()
            for rmId in stmToRemove:
                del ng.noteStemList[rmId]
            if len(ng.noteStemList) == 0:
                noteGroupList[ngId] = None
    else:
        imgrgb2 = imgrgb
    noteGroupMap = np.zeros((beamMapImg.shape[0], beamMapImg.shape[1]), dtype=np.uint16)
    ngImg = image.copy()

    colors = [(0,0,255),(0,255,0),(255,255,0), (0,120,230)]
    for idx,ng in enumerate(noteGroupList):
        if ng is not None:
            x0,y0,x1,y1 = ng.boundingBox
            noteGroupMap = cv2.rectangle(noteGroupMap, (x0,y0),(x1,y1), idx, -1)
            staffnum = np.max(mapStaffNum[y0:y1, x0:x1])
            mapStaffNum[y0:y1+1, x0:x1+1] = staffnum
            coloridx = -1 if staffnum ==0 else staffnum%3
            ngImg = cv2.rectangle(ngImg, (x0,y0),(x1,y1), colors[coloridx], 1, cv2.LINE_AA)
            for nb in ng.noteBoxes:
                xx0,yy0,xx1,yy1 = nb
                ngImg = cv2.rectangle(ngImg, (xx0,yy0),(xx1,yy1), colors[coloridx], 1, cv2.LINE_AA)
    beamMapImg = cv2.merge([mapBeamVal, middle, mapStaffNum]) 
    imwrite('ngImg.jpg',ngImg)
    imwrite('knnbeams.jpg',imgrgb)
    imwrite('knnbeams2.jpg',imgrgb2)
    return noteGroupList,  beamMapImg, noteGroupStemMap, noteGroupMap, {'knnbeams':imgrgb,'ngImg': ngImg, 'knnbeams2': imgrgb2}


def findRests(dataDict:dict, 
              beamMapImg: np.ndarray,barheight:float, 
              model: Rest_Classifier,):
    restList:List[Rest|None] = [None]
    restMap = np.zeros((beamMapImg.shape[0], beamMapImg.shape[1]), dtype=np.uint16)
    symbol = dataDict['symbols'].astype(np.uint8) # 1 where there's symbols
    stem_rests = dataDict['stems_rests'].astype(np.uint8)
    bb,gg,rr = cv2.split(beamMapImg)
    mask = np.ones_like(symbol, dtype= np.uint8)
    stemRestClean = cv2.cvtColor(stem_rests*255, cv2.COLOR_GRAY2BGR)
    ys,xs = np.where(bb==255)
    mask[ys,xs] = 0
    stem_rests[ys,xs] = 0
    # mask = cv2.erode(mask.astype(np.uint8), np.ones((barheight//2, barheight//4),dtype=np.uint8)).squeeze()
    ys,xs = np.where(gg>0)
    mask[ys,xs] = 0
    stem_rests[ys,xs] = 0
    stem_rests = cv2.dilate(stem_rests.astype(np.uint8), np.ones((barheight//3, barheight//3), dtype= np.uint8))

    # remaining = cv2.bitwise_or(cv2.bitwise_and(mask,symbol), stem_rests)*255
    remaining = cv2.bitwise_and(mask,symbol)*255
    remaining = cv2.cvtColor(remaining,cv2.COLOR_GRAY2BGR)
    symbol = cv2.cvtColor(symbol*255,cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(stem_rests.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    stem_rests = cv2.cvtColor((stem_rests*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    # save the image version (later can give img128 version), stem_rests itself, remaining
    bboxes = []
    longBoxes = []
    # imageClean = image.copy()
    remainingClean = remaining.copy()
    class_colors = [(255,0,0),(0,0,255),(255,255,0),(0,120,255),(40,255,40), (245, 220, 255),(230,130,175),(165,170, 70)]
    restClassNames = ['X', 'r16', 'r32', 'r4', 'r8'] 
    restToRhythm = [0,2,3,0,1]
    imgrgb = dataDict['image'].copy()
    for j in range(0,len(restClassNames)):
        imgrgb = cv2.putText(imgrgb,restClassNames[j], (30, 30+j*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_colors[j], 2, cv2.LINE_AA)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # filter out too big of boxed (likely some error)
        if (w>barheight*2 and h>barheight):
            continue
        if (w>barheight*0.6 and h>barheight):
            box = (max(x-barheight//2, 0), max(y-barheight*2, 0), min(x+w+barheight//2, symbol.shape[1]-1), min(y+h+barheight*2, symbol.shape[0]-1))
            # shrink the box so it centers around the rest
            yTopBlank = max(np.max(np.where(np.sum(remainingClean[box[1]:(box[1]+box[3])//2, box[0]:box[2],1],1)<255)[0].tolist()+[0])-barheight//4, 0)
            # the index of the ending (the new height)
            yBottomBlank = min(np.min(np.where(np.sum(remainingClean[(box[1]+box[3])//2:box[3], box[0]:box[2],1],1)<255)[0].tolist()+[box[3]-box[1]])+(box[3]-box[1])//2+barheight//4, box[3]-box[1])
            # xLeftBlank = max(np.max(np.where(np.sum(remainingClean[box[1]:box[3], box[0]:(box[0]+box[2])//2,1],0)<255)[0].tolist()+[0]), 0)
            # yRightBlank = min(np.min(np.where(np.sum(remainingClean[box[1]:box[3], (box[0]+box[2])//2:box[2],1],0)<255)[0].tolist()+[box[2]-box[0]])+(box[2]-box[0])//2, box[2]-box[0])
            # redo the training? No
            predict_idx = model.predict(remainingClean[box[1]:box[3], box[0]:box[2],:])
            # predict_idx = model.predict(remainingClean[box[1]+yTopBlank:box[1]+yBottomBlank, box[0]+xLeftBlank:box[0]+yRightBlank,:])
            box = (x, box[1]+yTopBlank, x+w, box[1]+yBottomBlank)
            if predict_idx>0:
                bboxes.append(box)
                restMap[box[1]:box[3],box[0]:box[2]] = len(restList)
                restList.append(Rest(box,restToRhythm[predict_idx]))
                gg = cv2.rectangle(gg, (box[0], box[1]), (box[2],box[3]),5, -1)
                imgrgb = cv2.rectangle(imgrgb,(box[0], box[1]), (box[2],box[3]), class_colors[predict_idx], 2, cv2.LINE_AA)
                # remaining = cv2.rectangle(remaining, (box[0], box[1]), (box[2],box[3]), (0,255,0), 2, cv2.LINE_AA)
                # stem_rests = cv2.rectangle(stem_rests, (box[0], box[1]), (box[2],box[3]), (0,255,0), 2, cv2.LINE_AA)
                # symbol = cv2.rectangle(symbol, (box[0], box[1]), (box[2],box[3]), (0,255,0), 2, cv2.LINE_AA)
        elif (w>barheight*0.8 and h<barheight*1.2 and h>barheight*0.3):
            box = (x,y,x+w,y+h)
            longBoxes.append(box)
            gg = cv2.rectangle(gg, (box[0], box[1]), (box[2],box[3]),5, -1)
            imgrgb = cv2.rectangle(imgrgb,(box[0], box[1]), (box[2],box[3]), (255,0,255), 2, cv2.LINE_AA)
            restMap[box[1]:box[3],box[0]:box[2]] = len(restList)
            restList.append(Rest(box,-1))
            # remaining = cv2.rectangle(remaining, (box[0], box[1]), (box[2],box[3]), (0,0,255), 2, cv2.LINE_AA)
            # stem_rests = cv2.rectangle(stem_rests, (box[0], box[1]), (box[2],box[3]), (0,0,255), 2, cv2.LINE_AA)
            # symbol = cv2.rectangle(symbol, (box[0], box[1]), (box[2],box[3]), (0,0,255), 2, cv2.LINE_AA)
    # init_rest_folder()            
    # labelRestData([imageClean, stemRestClean, remainingClean], bboxes, imgname)
    imwrite('remaining.jpg',remaining)
    imwrite('remainingStemRests.jpg',stem_rests)
    imwrite('RestClassification.jpg',imgrgb)
    beamMapImg = cv2.merge([bb,gg,rr])
    # return {'RestBarline1': remaining, 'RestBarline2': symbol}
    return restMap, restList, beamMapImg, {'RestClassification': imgrgb}

def mergeVerticalNoteGroups(staffObjList:List[Staff],noteGroupList:List[NoteGroup], noteGroupMap:np.ndarray,beamMapImg:np.ndarray, image:np.ndarray,
                            restMap, restList:List[Rest]):
    for staff in staffObjList:
        barh = staff.get_yOne()
        oneStep = barh//3
        y0 = max(staff.ys[0]-int(barh*2.5),0)
        y1 = min(staff.ys[-1]+int(barh*2.5), noteGroupMap.shape[0])
        tempRestMap = restMap.copy()
        for x in range(staff.left, staff.right-oneStep, oneStep):
            indexes = set(np.unique(noteGroupMap[y0:y1, x]))
            indexesNext = set(np.unique(noteGroupMap[y0:y1, x+oneStep]))
            rIndex = set(np.unique(tempRestMap[y0:y1, x]))
            rIndexNext = set(np.unique(tempRestMap[y0:y1, x+oneStep]))
            line = set(np.unique(beamMapImg[y0:y1, x:x+oneStep,2]))
            indexes -= {0}
            indexesNext-={0}
            line-={0}
            rIndex-={0}
            rIndexNext-={0}
            if len(line)>1:
                continue
            if len(indexes) == 0:
                continue
            # the overlapping x width is at least oneStep width
            noteGroupId = list(indexes)[0]
            lineList = list(line)
            idxList = list(indexes)

            if len(indexes)>1 and indexes==indexesNext: # two noteGroup overlap
                noteGroupId = idxList.pop()
                for gid in idxList:
                    ys,xs = np.where(noteGroupMap == gid)
                    noteGroupMap[ys,xs] = noteGroupId
                    beamMapImg[ys, xs, 2] = lineList[0]
                    noteGroupList[noteGroupId]=mergeNoteGroup(noteGroupList[noteGroupId], noteGroupList[gid], beamMapImg)
                    noteGroupList[gid] = None
            if len(rIndex)>0 and rIndex == rIndexNext:
                ridLst = list(rIndex)
                for rid in ridLst:
                    ys,xs = np.where(tempRestMap == rid)
                    if restList[rid].rhythm == -1:
                        continue
                    else:
                        tempRestMap[ys,xs] = 0
                        beamMapImg[ys,xs,2] = lineList[0]
                        noteGroupList[noteGroupId].addRest(restList[rid])
                        restList[rid].setNgId(noteGroupId)                
    ngImg = image.copy()
    colors = [(0,0,255),(0,255,0),(255,255,0), (0,120,230)]
    for ng in noteGroupList:
        if ng is not None:
            x0,y0,x1,y1 = ng.boundingBox
            ngImg = cv2.rectangle(ngImg, (x0,y0),(x1,y1), colors[(np.max(beamMapImg[y0:y1, x0:x1, 2])%3 if np.max(beamMapImg[y0:y1, x0:x1, 2])>0 else 3)], 2, cv2.LINE_AA)
    imwrite('vertically_merged.jpg', ngImg)
    return noteGroupList, noteGroupMap, beamMapImg, {'vertically_merged':ngImg}

def symbol_classification(dataDict: dict, model:Sfn_Clef_classifier, bar_height:int):
    def getbbox(clefs_keys: np.ndarray, bar_height:int)-> List[Tuple[int,int,int,int]]:
        contours, _ = cv2.findContours(clefs_keys.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w>3 and h>bar_height*1.6 and h>w:
                box = (x, y, x+w, y+h)
                bboxes.append(box)
        return bboxes
    def getBestRange(avg_color_lst: list, desire_length:int):
        if desire_length>=len(avg_color_lst)-1:
            return 0, len(avg_color_lst)-1
        most_white_value = 0
        most_white_ending_idx = 0
        for i in range(desire_length,len(avg_color_lst)-1):
            currSum = sum(avg_color_lst[i-desire_length:i])
            if currSum>most_white_value:
                most_white_value=currSum
                most_white_ending_idx = i
        return most_white_ending_idx-desire_length, most_white_ending_idx
    def sfnClefToDataType(bbox:Tuple[int,int,int,int], predictionId: int) -> Union[Accidentals, Clef]:
        # 'BassF', 'ViolaC', 'flat', 'natural', 'noClass', 'sharp', 'trebleG'
        isClef = [True, True, False, False, False, False, True]
        mapId = [-1, 0, -1, 0, -3, 1, 1]
        # we should never try to assign a noClass
        assert predictionId != 4
        if isClef[predictionId]:
            return Clef(bbox, mapId[predictionId])
        else:
            return Accidentals(bbox, mapId[predictionId])

    # output_dir will be images/tch/tch SYMBOL
    clefs_keys_ori:np.ndarray = dataDict['clefs_keys']
    clefs_keys = (clefs_keys_ori*255).astype(np.uint8)
    clefs_keys_expand = cv2.dilate(clefs_keys, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)))
    image:np.ndarray = dataDict['image']
    round1_sfn_img = image.copy()
    ori_bboxes = getbbox(clefs_keys_expand, bar_height)
    bboxes = merge_nearby_bbox(ori_bboxes, bar_height*3)
    class_names = ['BassF', 'ViolaC', 'flat', 'natural', 'noClass', 'sharp', 'trebleG','wide']  # Replace with actual class names
    class_symbols = ['Bass','Viola','b','n','x','#','Treble','Wide']
    class_colors = [(255,0,0),(0,255,0),(255,255,0),(255,0,125),(0,120,255),(255,0,255),(0,0,255),(101,134,168)]
    for j in range(0,len(class_colors)):
        round1_sfn_img = cv2.putText(round1_sfn_img,class_names[j], (30, 30+j*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_colors[j], 2, cv2.LINE_AA)
    noClass_bbox = []
    sfn_height_lst = []
    sfn_width_lst = []
    # the 0th is a stub for better alignment
    returnSfnList:List[Union[Accidentals,Clef, None]] = [None]
    for b in bboxes:
        if (b[3]-b[1])<(b[2]-b[0]): # height<width: wide
            crop_img = clefs_keys[b[1]:b[3], b[0]:b[2]]
            predict_idx = model.predictWide(crop_img)
            if predict_idx ==4:
                predict_idx = 7
        else:
            crop_img = clefs_keys[b[1]:b[3], b[0]:b[2]]
            predict_idx = model.predict(crop_img)
            if predict_idx == 0 and b[3]-b[1]<bar_height*2: # if it's Bass but really small -> likely it's flat instead
                predict_idx = 2
            if predict_idx == 2 or predict_idx==5 or predict_idx==3:
                # this is flat | sharp | natural
                sfn_height_lst.append(b[3]-b[1])
                sfn_width_lst.append(b[2]-b[0])
        round1_sfn_img = cv2.rectangle(round1_sfn_img, (b[0],b[1]),(b[2],b[3]), class_colors[predict_idx],2,cv2.LINE_AA)
        round1_sfn_img = cv2.putText(round1_sfn_img, class_symbols[predict_idx], (b[0],b[3]+10),cv2.FONT_HERSHEY_SIMPLEX,  1, class_colors[predict_idx], 1, cv2.LINE_AA)
        if predict_idx == 4 or predict_idx==7:
            noClass_bbox.append(b)
        else:
            returnSfnList.append(sfnClefToDataType(b, predict_idx))
    noClass_bbox2 = []
    round2_sfn_img = round1_sfn_img.copy()
    sfn_median_height = bar_height*2
    sfn_median_width = bar_height
    if len(sfn_height_lst)>0:
        sfn_median_height = int(np.median(sfn_height_lst))
        sfn_median_width = int(np.median(sfn_width_lst))
    
    for box1 in noClass_bbox:
        x0 = box1[0]
        y0 = box1[1]
        clef_key_crop = clefs_keys[box1[1]:box1[3], box1[0]:box1[2]]
        smallbbox = getbbox(clef_key_crop, bar_height)
        if len(smallbbox)<=1:
            if len(smallbbox)==1 and ((box1[2]-box1[0])>sfn_median_width*1.5 or (box1[3]-box1[1])>sfn_median_height*1.5):
                noClass_bbox2.append(box1)
            predict_idx = 4
            round2_sfn_img = cv2.rectangle(round2_sfn_img, (box1[0],box1[1]),(box1[2],box1[3]), class_colors[predict_idx],2,cv2.LINE_AA)
            round2_sfn_img = cv2.putText(round2_sfn_img, class_symbols[predict_idx], (box1[0],box1[3]+10),cv2.FONT_HERSHEY_SIMPLEX,  1, class_colors[predict_idx], 1, cv2.LINE_AA)
            continue
        for bb in smallbbox:
            b = [x0+bb[0],y0+bb[1],x0+bb[2],y0+bb[3]]
            small_crop = clefs_keys[b[1]:b[3], b[0]:b[2]]
            predict_idx = model.predict(small_crop)
            round2_sfn_img = cv2.rectangle(round2_sfn_img, (b[0],b[1]),(b[2],b[3]), class_colors[predict_idx],2,cv2.LINE_AA)
            round2_sfn_img = cv2.putText(round2_sfn_img, class_symbols[predict_idx], (b[0],b[3]+10),cv2.FONT_HERSHEY_SIMPLEX,  1, class_colors[predict_idx], 1, cv2.LINE_AA)

            if predict_idx == 4:
                if (small_crop.shape[0]>sfn_median_height*1.5 and small_crop.shape[1]>sfn_median_width*0.8) or (small_crop.shape[0]>sfn_median_height*0.8 and small_crop.shape[1]>sfn_median_width*1.5):
                    noClass_bbox2.append(b)
                else:
                    continue
            else:
                returnSfnList.append(sfnClefToDataType(b, predict_idx))
    for box1 in noClass_bbox2:
        x0 = box1[0]
        y0 = box1[1]
        w = box1[2]-x0
        h = box1[3]-y0
        seg_img = clefs_keys[y0:y0+h, x0:x0+w]
        width_ratio =w/sfn_median_width
        height_ratio = h/sfn_median_height
        num_symbol = round(max(width_ratio, height_ratio))
        if num_symbol<2:
            continue
        if width_ratio>height_ratio:
            left_center = sfn_median_width//2
            right_center = w-sfn_median_width//2
            avg_width:float = (right_center-left_center)/(num_symbol-1)
            for i in range(num_symbol):
                img_left = int(i*avg_width)
                crop_img = seg_img[:,img_left:min(img_left+sfn_median_width,seg_img.shape[1])]
                vertical_avg = np.mean(crop_img,1)
                front,back = getBestRange(vertical_avg, sfn_median_height)
                # b: x0, y0, x1, y1 relative to the croped image
                b = [max(img_left,0),max(front,0),
                     min(img_left+sfn_median_width,seg_img.shape[1]), min(back,seg_img.shape[0])]
                crop_crop_img = seg_img[b[1]:b[3], b[0]:b[2]]
                # predict_idx = model.predict(crop_crop_img)
                if sum(np.max(crop_crop_img,1))<=0:
                    continue
                if sum(np.max(crop_crop_img,1))/255/crop_crop_img.shape[0]<=0.6:
                    continue
                predict_lst = model.get_prediction_vector(crop_crop_img)
                pred_filter = [x for x in predict_lst if x not in [0,1,4,6,7]]
                predict_idx = pred_filter[0]
                returnSfnList.append(sfnClefToDataType((x0+b[0], y0+b[1], x0+b[2],y0+b[3]), predict_idx))
        else:
            seg_img_to_delete = seg_img.copy()
            vertical_boxes = []
            pred_idx = []

            b = [0,0,w,sfn_median_height] #x0, y0, x1, y1
            hor_avg_top = np.mean(seg_img[b[1]:b[3],:],0)
            b[0],b[2] = getBestRange(hor_avg_top, sfn_median_width)
            seg_img_to_delete[b[1]:b[3],b[0]:b[2]] = 0
            vertical_boxes.append(b)

            b = [0,h-sfn_median_height-1,w,h-1]
            hor_avg_bot = np.mean(seg_img[b[1]:b[3],:],0)
            b[0],b[2] = getBestRange(hor_avg_bot, sfn_median_width)
            seg_img_to_delete[b[1]:b[3],b[0]:b[2]] = 0
            vertical_boxes.append(b)

            if h>sfn_median_height*2:
                b = [0,0,0,0]
                if np.max(seg_img_to_delete) == 0: # if there's some issue with the bounding box basically
                    break
                # middle_del = seg_img_to_delete[sfn_median_height//2:h-sfn_median_height//2,:]
                ver_avg_mid = np.mean(seg_img_to_delete,1)
                b[1],b[3] = getBestRange(ver_avg_mid, sfn_median_height)
                expanded_left_right = [min(vertical_boxes[0][0],vertical_boxes[1][0]),
                                       max(vertical_boxes[0][2],vertical_boxes[1][2])]
                if expanded_left_right[0]>=w-expanded_left_right[1]-sfn_median_width//5:
                    b[2] = sfn_median_width
                else:
                    b[0] = w-sfn_median_width-1
                    b[2] = w-1
                vertical_boxes.append(b)
                seg_img_bgr = cv2.cvtColor(seg_img,cv2.COLOR_GRAY2BGR)
                seg_img_bgr = cv2.rectangle(seg_img_bgr, (b[0],b[1]),(b[2],b[3]),(0,255,0),2,cv2.LINE_AA)

            for idx, b in enumerate(vertical_boxes):
                predict_lst = model.get_prediction_vector(seg_img[b[1]:b[3],b[0]:b[2]])
                pred_filter = [x for x in predict_lst if x not in [0,1,4,6,7]]
                predict_idx = pred_filter[0]
                if idx == 2 and predict_lst[0] == 4:
                    continue
                returnSfnList.append(sfnClefToDataType((x0+b[0], y0+b[1], x0+b[2],y0+b[3]), predict_idx))
    sfnClefImg = image.copy()
    class_colors2 = [(255,255,0),(255,0,125),(255,0,255),(255,0,0),(0,255,0),(0,0,255)]
    class_names2 = ['flat', 'natural', 'sharp','BassF', 'ViolaC', 'trebleG']
    for j in range(0,len(class_colors2)):
        sfnClefImg = cv2.putText(sfnClefImg,class_names2[j], (30, 30+j*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_colors2[j], 2, cv2.LINE_AA)
    # class_names = ['BassF', 'ViolaC', 'flat', 'natural', 'noClass', 'sharp', 'trebleG','wide']
    
    sfnClefMap = np.zeros((sfnClefImg.shape[0], sfnClefImg.shape[1]), dtype=np.uint16)
    for idx,clefAccidentals in enumerate(returnSfnList):
        if clefAccidentals is not None:
            x0,y0,x1,y1 = clefAccidentals.getBbox()
            # 0 if it's accidentals
            predict_idx = clefAccidentals.getType()*3+clefAccidentals.getValue()+1
            sfnClefImg = cv2.rectangle(sfnClefImg, (x0,y0),(x1,y1), class_colors2[predict_idx],2,cv2.LINE_AA)
            sfnClefMap = cv2.rectangle(sfnClefMap, (x0,y0),(x1,y1), idx, -1)

    imwrite('round1Sfn.jpg',round1_sfn_img)
    imwrite('round2Sfn.jpg',round2_sfn_img)
    imwrite('round3Sfn.jpg', sfnClefImg)
    
    return returnSfnList, sfnClefMap, {'sfnClef':sfnClefImg}

def exportYolo(sfnClefList:List[Accidentals|Clef], image:np.ndarray,img_name:str, barheight: int):
    if not os.path.isdir("yolo"):
        os.mkdir("yolo")
    if not os.path.isdir("yolo/debug"):
        os.mkdir("yolo/debug")
    height, width, _ = image.shape
    outputString = ""
    imgg = image.copy()
    for sfnClef in sfnClefList:
        if type(sfnClef) == Clef:
            if sfnClef.getValue() == 0 or sfnClef.getValue() == -2:
                x0,y0,x1,y1 = sfnClef.boundingBox
                if y1-x1 < 3*barheight:
                    continue
                cv2.rectangle(imgg, (x0,y0),(x1,y1),(0,0,255), 2, cv2.LINE_AA)
                xPerc = (x1+x0)/2/width
                yPerc = (y1+y0)/2/height
                xWidth = (x1-x0)/width
                yHeight = (y1-y0)/height
                print(f"{1} {xPerc:.6f} {yPerc:.6f} {xWidth:.6f} {yHeight:.6f}")
                outputString += f"{1} {xPerc:.6f} {yPerc:.6f} {xWidth:.6f} {yHeight:.6f}\n"
        elif type(sfnClef) == Accidentals:
            if sfnClef.getValue() == -1:
                # is flat
                x0,y0,x1,y1 = sfnClef.boundingBox
                cv2.rectangle(imgg, (x0,y0),(x1,y1),(0,255,0), 2, cv2.LINE_AA)
                xPerc = (x1+x0)/2/width
                yPerc = (y1+y0)/2/height
                xWidth = (x1-x0)/width
                yHeight = (y1-y0)/height
                print(f"{0} {xPerc:.6f} {yPerc:.6f} {xWidth:.6f} {yHeight:.6f}")
                outputString += f"{0} {xPerc:.6f} {yPerc:.6f} {xWidth:.6f} {yHeight:.6f}\n"
    with open(f"yolo/{img_name}.txt","w") as f:
        f.write(outputString)
    cv2.imwrite(f"yolo/{img_name}.jpg", image)
    cv2.imwrite(f"yolo/debug/{img_name}.jpg",imgg)

# filter out the sfn that is overlapping with noteGroupList
def filterSfnClefModifyBeamMap(sfnClefList:List[Union[Accidentals, Clef, None]], 
                               noteGroupMap: np.ndarray, 
                               beamMapImg:np.ndarray):
    sfnClefMap = np.zeros((beamMapImg.shape[0], beamMapImg.shape[1]), dtype=np.uint16)
    bb,gg,rr = cv2.split(beamMapImg)
    assert len(np.unique(gg)) <=2
    for idx, sfnClef in enumerate(sfnClefList):
        if sfnClef is not None:
            x0,y0,x1,y1 = sfnClef.getBbox()
            if np.sum(noteGroupMap[y0:y1,x0:x1]>0)/((x1-x0)*(y1-y0))>0.5:
                sfnClefList[idx] = None
            else:
                sfnClefMap[y0:y1,x0:x1] = idx
                # 2 for accidentals, 3 for clef
                gg[y0:y1,x0:x1] = sfnClef.getType()+2
    # will want the noteGroup to be "more dominent" than the sfnClef, so it overwrites if it's place of a sfn
    ys,xs = np.where(noteGroupMap>0)
    gg[ys,xs] = 1
    beamMapImg = cv2.merge([bb,gg,rr])
    return sfnClefList,sfnClefMap, beamMapImg

def enhanceStaff(image:np.ndarray, staffObjList:List[Staff],barheight:int):
    _, img200 = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    for staff in staffObjList:
        if staff.IsStaffAligned(barheight):
            for y in staff.ys:
                img200 = cv2.line(img200, (staff.left, y), (staff.right, y), (0,0,0), 1, cv2.LINE_AA)
    imwrite('enhanceStaff.jpg',img200)
    return img200

def assignPitch(image:np.ndarray, noteGroupList:List[NoteGroup|None], beamMapImg:np.ndarray, barheight:int):
    # assign the "pitch" relative to the center of the staff (12.5),
    # ex: in violin, B will be 0, C:1,D:2, A: -1 etc
    # if it doesn't have a staff line id yet, group it using the beamMapImg[:,:,4]
    # image is for outputting debug images
    pitchImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, imgBin = cv2.threshold(pitchImg, 200, 1, cv2.THRESH_BINARY_INV)
    _,pitchImg = cv2.threshold(pitchImg, 200, 255, cv2.THRESH_BINARY)
    pitchLineImg = pitchImg.copy()
    pitchLineImg = cv2.cvtColor(pitchLineImg, cv2.COLOR_GRAY2BGR)
    pitchImg = pitchImg//4+191
    pitchImg = cv2.cvtColor(pitchImg, cv2.COLOR_GRAY2BGR)
    pitch_colors = [(255,0,0),(0,0,255),(255,255,0),(0,120,255),(40,255,40), (255,0,255),(230,130,175),(165,170, 70)]
    pitch_names = ['B','C','D','E','F','G','A']
    for j in range(0,len(pitch_names)):
        pitchImg = cv2.putText(pitchImg,pitch_names[j], (30, 30+j*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pitch_colors[j], 2, cv2.LINE_AA)
    for ng in noteGroupList:
        if ng is None:
            continue
        for stemObj in ng.noteStemList:
            x0,y0,x1,y1 = stemObj.smallNoteBox
            staffNum = np.max(beamMapImg[y0:y1,x0:x1,2])
            if staffNum == 0:
                staffNum = round(np.sum(beamMapImg[y0:y1,4,2]/(y1-y0)))
            height = np.sum((beamMapImg[y0:y1,staffNum%4,2]-12.5)*2)/(y1-y0)
            stemObj.pitchFloat = height

            x0,y0,x1,y1 = stemObj.noteBox
            height2 = np.sum((beamMapImg[y0:y1,staffNum%4,2]-12.5)*2)/(y1-y0)
            if height2-int(height2) == 0:
                stemObj.pitchWideFloat = height
            else:
                stemObj.pitchWideFloat = height2

    pitchAndLineImg = pitchImg.copy()
    modifiedPitchImg = pitchImg.copy()

    for ngIdx, ng in enumerate(noteGroupList):
        if ng is None:
            continue
        for idx,stemObj in enumerate(ng.noteStemList):
            x0,y0,x1,y1 = stemObj.smallNoteBox
            _,y0B, _, y1B = stemObj.noteBox
            pitch = stemObj.pitchFloat
            xx0 = max(x0-barheight//2, 0)
            xx1 = min(x1+barheight//2, beamMapImg.shape[1]-1)
            cc = np.sum(imgBin[max(y0B,y0-2):min(y1B,y1+3), xx0:xx1],1).tolist()
            ccNozero = [c for c in cc if c>0]
            maxIdx = [i for i in range(len(ccNozero)) if ccNozero[i]==max(ccNozero)]
            # center that was tilted and thick
            if (maxIdx[0]>len(ccNozero)/2 and maxIdx[0]<len(ccNozero)*0.)or (maxIdx[-1]<len(ccNozero)/2 and maxIdx[-1]>len(ccNozero)*0.8) and maxIdx[-1]-maxIdx[0]>=3: # TODO
                stemObj.hasLineMiddle = True
                pitchLineImg=cv2.rectangle(pitchLineImg,(xx0, y0+maxIdx[0]),(xx1, y0+maxIdx[-1]),(255,0,255), 2, cv2.LINE_AA)
            # center
            if len(set.intersection(set([len(ccNozero)//2,len(ccNozero)//2-1, len(ccNozero)//2+1]), maxIdx))>0:
                stemObj.hasLineMiddle = True
                pitchLineImg=cv2.rectangle(pitchLineImg,(xx0, y0+maxIdx[0]),(xx1, y0+maxIdx[-1]),(255,0,255), 2, cv2.LINE_AA)
            # top or bottom
            elif y1-y0>barheight//2 and (maxIdx[0]<=3 or maxIdx[-1]>=len(ccNozero)-4):
                stemObj.hasLineMiddle = False
                pitchLineImg=cv2.rectangle(pitchLineImg,(xx0, y0),(xx1, y1),(0,255,0), 1, cv2.LINE_AA)
            # if it has line in the middle
            elif ccNozero[maxIdx[0]]>(x1-x0):
                stemObj.hasLineMiddle = True
                pitchLineImg=cv2.rectangle(pitchLineImg,(xx0, y0+maxIdx[0]),(xx1, y0+maxIdx[-1]),(255,0,255), 2, cv2.LINE_AA)
            else:
                stemObj.hasLineMiddle = True
                pitchLineImg=cv2.rectangle(pitchLineImg,(xx0, y0+maxIdx[0]),(xx1, y0+maxIdx[-1]),(255,255,0), 2, cv2.LINE_AA)

            p1 = math.ceil(pitch)
            p2 = math.floor(pitch)
            pitchLineImg = cv2.putText(pitchLineImg, str(ngIdx), (x0,y0-barheight*2), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,0), 1, cv2.LINE_AA)
            pitchAndLineImg = cv2.putText(pitchAndLineImg, str(ngIdx), (x0,y0-barheight*2), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,0), 1, cv2.LINE_AA)
            pitchAndLineImg = cv2.putText(pitchAndLineImg, str(pitch_names[p1%7]), (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pitch_colors[p1%7], 2, cv2.LINE_AA)
            pitchAndLineImg = cv2.putText(pitchAndLineImg, str(pitch_names[p2%7]), (x0,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pitch_colors[p2%7], 2, cv2.LINE_AA)
            pitchAndLineImg = cv2.rectangle(pitchAndLineImg, (x0,y0),(x1,y1), (30,30,30), 1,cv2.LINE_AA)
            pitchImg = cv2.putText(pitchImg, str(pitch_names[round(pitch)%7]), (x0,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pitch_colors[round(pitch)%7], 2, cv2.LINE_AA)
            pitchImg = cv2.rectangle(pitchImg, (x0,y0),(x1,y1), (30,30,30), 1,cv2.LINE_AA)
            
    xend = pitchLineImg.shape[1]-1
    for i in range(pitchLineImg.shape[0]-1):
        currStaffNum = beamMapImg[i,4,2]
        if beamMapImg[i,currStaffNum%4, 2] != beamMapImg[i+1,currStaffNum%4, 2]:
            pitchLineImg = cv2.line(pitchLineImg,(0,i),(xend,i),pitch_colors[currStaffNum%4],thickness=1, lineType=cv2.LINE_AA)
            pitchAndLineImg = cv2.line(pitchAndLineImg,(0,i),(xend,i),pitch_colors[currStaffNum%4],thickness=1, lineType=cv2.LINE_AA)
    
    for idx, ng in enumerate(noteGroupList):
        if ng is None:
            continue
        for stem in ng.noteStemList:
            pitchInt = stem.getBestPitchInt()
            x0,y0,x1,y1 = stem.noteBox
            # modifiedPitchImg = cv2.putText(modifiedPitchImg, str(idx), (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pitch_colors[pitchInt%7], 2, cv2.LINE_AA)
            modifiedPitchImg = cv2.putText(modifiedPitchImg, str(pitch_names[pitchInt%7]), (x0,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pitch_colors[pitchInt%7], 2, cv2.LINE_AA)
            modifiedPitchImg = cv2.rectangle(modifiedPitchImg, (x0,y0),(x1,y1), (30,30,30), 1,cv2.LINE_AA)
            stem.pitchSoprano = pitchInt
        
    
    imwrite('pitchline.jpg', pitchLineImg)
    imwrite('pitchAndLine.jpg', pitchAndLineImg)
    imwrite('pitch.jpg', pitchImg)
    imwrite('pitchModified.jpg', modifiedPitchImg)
    return noteGroupList, {'pitch':pitchImg, 
                           'pitchAndLine':pitchAndLineImg, 
                           'pitchline': pitchLineImg,
                           'pitchModified':modifiedPitchImg}

def getNoteChunks(image: np.ndarray, noteGroupMap:np.ndarray, beamMapImg:np.ndarray, noteGroupVerticallyMerged:List[NoteGroup]):
    imgrgb = image.copy()
    beamBinary = np.zeros((beamMapImg.shape[0],beamMapImg.shape[1]))
    xs,ys = np.where(beamMapImg[:,:,0]==255)
    beamBinary[xs,ys] = 1
    contours, _ = cv2.findContours(beamBinary.astype(np.uint8),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    beamBoxes:List[Tuple[int,int,int,int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        beamBoxes.append((x,y,x+w,y+h))
        imgrgb = cv2.rectangle(imgrgb,(x,y), (x+w, y+h), (0,0,255), 2, cv2.LINE_AA)
    imwrite('beamContour.jpg',imgrgb)

    noteChunkList:List[NoteChunk] = [None]
    noteChunkMap = np.zeros_like(noteGroupMap)
    for box in beamBoxes:
        currentNoteChunkIdxs = set(np.unique(noteChunkMap[box[1]:box[3], box[0]:box[2]]))
        allNoteGroupIdxs = set(np.unique(noteGroupMap[box[1]:box[3], box[0]:box[2]]))
        allNoteGroupIdxs.add(0)
        allNoteGroupIdxs.remove(0)
        currentNoteChunkIdxs.add(0)
        currentNoteChunkIdxs.remove(0)
        if len(allNoteGroupIdxs) == 0:
            continue
        if len(currentNoteChunkIdxs)==0:
            n = NoteChunk(allNoteGroupIdxs)
            for a in allNoteGroupIdxs:
                ys,xs = np.where(noteGroupMap==a)
                noteChunkMap[ys,xs] = len(noteChunkList)
            noteChunkList.append(n)
        elif len(currentNoteChunkIdxs) == 1:
            noteChunkList[min(currentNoteChunkIdxs)].mergeNoteChunk(allNoteGroupIdxs)
            for a in allNoteGroupIdxs:
                ys,xs = np.where(noteGroupMap==a)
                noteChunkMap[ys,xs] = min(currentNoteChunkIdxs)
        else:
            minChunkIdx = min(currentNoteChunkIdxs)
            currentNoteChunkIdxs.remove(minChunkIdx)
            for chunkIdx in list(currentNoteChunkIdxs):
                ys,xs = np.where(noteChunkMap==chunkIdx)
                noteChunkMap[ys,xs] = minChunkIdx
                noteChunkList[minChunkIdx].mergeNoteChunk(noteChunkList[chunkIdx].noteGroupIdxs)
                noteChunkList[chunkIdx] = None
    imggg = image.copy()
    for ii, nc in enumerate(noteChunkList):
        if nc is None:
            continue
        for id in nc.noteGroupIdxs:
            noteGroupVerticallyMerged[id].noteChunkId = ii
        bx = nc.getBoundingBox(noteGroupVerticallyMerged) # changed here
        imggg = cv2.rectangle(imggg, (bx[0],bx[1]),(bx[2], bx[3]),(0,0,255), 2, cv2.LINE_AA)
    imwrite("horizontalGrouping.jpg", imggg)
    return noteChunkList, imggg
    
def findBarlines(image:np.ndarray, beamMapImg:np.ndarray, staffObjList:List[Staff], noteGroupStemMap:np.ndarray, thres:float)->np.ndarray:
    img200 =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    barlineImg = np.zeros((image.shape[0], image.shape[1]),dtype=np.uint8)
    # (1) where there's stuff, (0) where there's no
    _,img200bin = cv2.threshold(img200, 200, 1, cv2.THRESH_BINARY_INV) 
    ys,xs = np.where(beamMapImg[:,:,1]>0)
    img200bin[ys,xs] = 0
    ys,xs = np.where(noteGroupStemMap>0)
    img200bin[ys,xs] = 0
    for idx,staff in enumerate(staffObjList):
        ys = staff.ys
        y0 = ys[0]
        y1 = ys[-1]
        xs = np.where(np.sum(img200bin[y0:y1, :],0)>(y1-y0)*thres)[0]
        beamMapImg[y0:y1,xs,1] = 4
        beamMapImg[y0:y1,xs-1,1] = 4
        beamMapImg[y0:y1,xs+1,1] = 4
        barlineImg[y0:y1, xs] = 1
    contours, _ = cv2.findContours(barlineImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    barlineboxes:List[Tuple[int,int,int,int]] = []
    imgbgr = image.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        barlineboxes.append((x,y,x+w,y+h))
        imgbgr = cv2.rectangle(imgbgr, (x,y), (x+w, y+h), (0,0,255),1,cv2.LINE_AA)
    imwrite("barLineImg.jpg",imgbgr)
    return beamMapImg, barlineboxes, imgbgr

def assignDots(noteGroupVerticallyMerged:List[NoteGroup],beamMapImg:np.ndarray, image:np.ndarray,bar_height:int,staffObjList:List[Staff]):
    imgbgr = image.copy() #bgr
    _,gg,_ = cv2.split(beamMapImg)
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = bar_height*bar_height//16
    params.maxArea = bar_height*bar_height//2

    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = True
    params.minInertiaRatio = 0.5
    
    detector = cv2.SimpleBlobDetector_create(params)
    _, thresh_image = cv2.threshold(image[:,:,1], 150, 255, cv2.THRESH_BINARY)
    thresh_origin = thresh_image.copy()
    for sf in staffObjList:
        for yOne in sf.ys:
            thresh_image[yOne-math.ceil(bar_height/8):yOne+bar_height//8+1,:] = 255

    for ng in noteGroupVerticallyMerged:
        if ng is None:
            continue
        for sm in ng.noteStemList:
            oldbox = sm.noteBox
            newbox = (oldbox[2],max(0,oldbox[1]-2*bar_height),
                      min(oldbox[2]+bar_height, image.shape[1]-1), min(oldbox[1]+2*bar_height, image.shape[0]-1))
            realWidth = np.max(np.append(np.where(np.max(gg[newbox[1]:newbox[3],newbox[0]:newbox[2]],0)==0)[0],0))
            # since the barline usually isn't thick enough
            realWidth = min(realWidth, np.min(np.append(np.where(np.max(gg[newbox[1]:newbox[3],newbox[0]:newbox[2]],0)==4)[0],realWidth)))
            if realWidth>=bar_height-1:
                realWidth = np.max(np.append(np.where(np.max(gg[newbox[1]:newbox[3],newbox[0]:min(newbox[2]+bar_height,image.shape[1]-1)],0)==0)[0],0))
                realWidth = min(realWidth, np.min(np.append(np.where(np.max(gg[newbox[1]:newbox[3],newbox[0]:min(newbox[2]+bar_height,image.shape[1]-1)],0)==4)[0],realWidth)))
            if realWidth>bar_height*0.5:
                dotBox = (oldbox[2], max(0,oldbox[1]-bar_height),
                        min(oldbox[2]+realWidth,image.shape[1]-1),min(oldbox[1]+int(bar_height*1.5), image.shape[0]-1))
                keypoints = detector.detect(thresh_image[dotBox[1]:dotBox[3], dotBox[0]:dotBox[2]])
                if len(keypoints)==0:
                    imgbgr = cv2.rectangle(imgbgr,(dotBox[0],dotBox[1]),(dotBox[2],dotBox[3]),(0,255,255),1,cv2.LINE_AA)
                elif (np.where(np.min(thresh_origin[dotBox[1]:dotBox[3], dotBox[0]+bar_height//2:dotBox[2]],1)==255)[0]).shape[0]>0 and np.min(np.where(np.min(thresh_origin[dotBox[1]:dotBox[3], dotBox[0]+bar_height//2:dotBox[2]],1)==255))>bar_height:
                    imgbgr = cv2.rectangle(imgbgr,(dotBox[0],dotBox[1]),(dotBox[2],dotBox[3]),(160,100,240),2,cv2.LINE_AA)
                else:
                    sm.hasdot = True
                    imgbgr = cv2.rectangle(imgbgr,(dotBox[0],dotBox[1]),(dotBox[2],dotBox[3]),(255,255,0),2,cv2.LINE_AA)
    imwrite('dotBox.jpg',imgbgr)
    return noteGroupVerticallyMerged,{'dotBox':imgbgr}

def assignRestDots(restList:List[Rest], beamMapImg:np.ndarray, noteGroupStemMap:np.ndarray, image:np.ndarray, bar_height:int, staffObjList:List[Staff]):
    imgbgr = image.copy() #bgr
    _,gg,_ = cv2.split(beamMapImg)
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = bar_height*bar_height//16
    params.maxArea = bar_height*bar_height//2

    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = True
    params.minInertiaRatio = 0.5
    
    detector = cv2.SimpleBlobDetector_create(params)
    _, thresh_image = cv2.threshold(image[:,:,1], 150, 255, cv2.THRESH_BINARY)
    thresh_origin = thresh_image.copy()
    for sf in staffObjList:
        for yOne in sf.ys:
            thresh_image[yOne-math.ceil(bar_height/8):yOne+bar_height//8+1,:] = 255
    for rs in restList:
        if rs is None:
            continue
        elif rs.rhythm<0:
            continue
        newbox = (rs.boundingBox[2],rs.boundingBox[1],
                    min(rs.boundingBox[2]+bar_height,image.shape[1]),min(rs.boundingBox[1]+2*bar_height,image.shape[0]))
        # 先用一個barheight寬度，高度從最高點往下兩個barheight，往上一個，用stem跟rest來縮減
        realWidth = np.max(np.append(np.where(np.max(noteGroupStemMap[newbox[1]:newbox[3],newbox[0]:newbox[2]],0)==0)[0],0))
        realWidth = min(realWidth, np.max(np.append(np.where(np.max(beamMapImg[newbox[1]:newbox[3],newbox[0]:newbox[2]],0)!=5)[0],0)))
        realWidth = min(realWidth, np.min(np.append(np.where(np.max(gg[newbox[1]:newbox[3],newbox[0]:newbox[2]],0)==4)[0],realWidth)))
        # 如果寬度還是滿格的話再往右延長到2*barheight用beamMapImg 來縮減
        if realWidth>=bar_height-1:
            realWidth = np.max(np.append(np.where(np.max(gg[newbox[1]:newbox[3],newbox[0]:min(newbox[2]+bar_height,image.shape[1]-1)],0)==0)[0],0))
            realWidth = min(realWidth, np.min(np.append(np.where(np.max(gg[newbox[1]:newbox[3],newbox[0]:min(newbox[2]+bar_height,image.shape[1]-1)],0)==4)[0],realWidth)))
        elif realWidth>bar_height*0.5:
            realWidth = realWidth+bar_height//3
        if realWidth>bar_height*0.5:
            dotBox = newbox
            keypoints = detector.detect(thresh_image[dotBox[1]:dotBox[3], dotBox[0]:dotBox[2]])
            if len(keypoints)==0:
                imgbgr = cv2.rectangle(imgbgr,(dotBox[0],dotBox[1]),(dotBox[2],dotBox[3]),(0,255,200),1,cv2.LINE_AA)
            elif len(np.where(np.min(thresh_origin[dotBox[1]:dotBox[3], dotBox[0]+bar_height//2:dotBox[2]],1)==255)[0]) == 0:
                imgbgr = cv2.rectangle(imgbgr,(dotBox[0],dotBox[1]),(dotBox[2],dotBox[3]),(255,255,0),2,cv2.LINE_AA)
            elif np.min(np.where(np.min(thresh_origin[dotBox[1]:dotBox[3], dotBox[0]+bar_height//2:dotBox[2]],1)==255))>bar_height:
                imgbgr = cv2.rectangle(imgbgr,(dotBox[0],dotBox[1]),(dotBox[2],dotBox[3]),(160,100,240),2,cv2.LINE_AA)
            else:
                imgbgr = cv2.rectangle(imgbgr,(dotBox[0],dotBox[1]),(dotBox[2],dotBox[3]),(255,255,0),2,cv2.LINE_AA)
                rs.hasdot = True
    imwrite('dotRestBox.jpg',imgbgr)
    return restList,{'dotRestBox':imgbgr}

def assignSfnToNote(image:np.ndarray, noteGroupMap:np.ndarray, noteGroupVerticallyMerged:List[NoteGroup],sfnClefMap:np.ndarray,sfnClefList:List[Union[Accidentals, Clef, None]]):
    stemIdxMap = np.ones((noteGroupMap.shape[0], noteGroupMap.shape[1]), dtype=np.uint8)*(-1)
    accidentalsImg = image.copy()
    for ng in noteGroupVerticallyMerged:
        if ng is None:
            continue
        for idx, stem in enumerate(ng.noteStemList):
            x0,y0,x1,y1 = stem.noteBox
            stemIdxMap[y0:y1, x0:x1] = idx
    sfnGroupList = []
    for currSfnIdx, sfnc in enumerate(sfnClefList):
        if sfnc is None:
            continue
        if sfnc.getType() == 1: # is Clef
            continue
        sfnc:Accidentals
        x0,y_0,x1,y_1 = sfnc.getBbox() # y0: smaller (top), y1: bigger (bottom)
        if sfnc.shift >=0: # sharp or natural
            y0 = (y_0+y_1)//2-(y_1-y_0)//6
            y1 = (y_0+y_1)//2+(y_1-y_0)//6
        else:
            y0 = (y_0+y_1)//2
            y1 = y_1
        sfnc.shrinkYs = (y0,y1)
        stemIdxLst = np.unique(stemIdxMap[y0:y1,x1:x1+(x1-x0)]).tolist()
        if -1 in stemIdxLst: 
            stemIdxLst.remove(-1)
        ngIdxLst = np.unique(noteGroupMap[y0:y1,x1:x1+(x1-x0)]).tolist()
        if 0 in ngIdxLst:
            ngIdxLst.remove(0)
        if len(stemIdxLst)==0 or len(ngIdxLst)==0:
            # is Key signature or has overlapping sfns
            sfnList = np.unique(sfnClefMap[y_0:y_1,x1:x1+(x1-x0)//2]).tolist()
            if sfnList == [0]:
                sfnc.isKeySignature = True
            else:
                if 0 in sfnList:
                    sfnList.remove(0)
                sfnNext = sfnList[0] # sfnNext is the index of the next sfn
                if type(sfnClefList[sfnNext]) is not Accidentals:
                    continue
                hasAssigned = False
                for sIdx in range(len(sfnGroupList)):
                    if sfnNext in sfnGroupList[sIdx]:
                        sfnGroupList[sIdx] = [currSfnIdx]+sfnGroupList[sIdx]
                        hasAssigned = True
                        continue
                    elif currSfnIdx in sfnGroupList[sIdx]:
                        sfnGroupList[sIdx] = sfnGroupList[sIdx]+[sfnNext]
                        hasAssigned = True
                        continue
                if not hasAssigned:
                    sfnGroupList.append([currSfnIdx, sfnNext])
            continue
        stemIdx = stemIdxLst[0]
        ngIdx = ngIdxLst[0]
        if len(ngIdxLst)>1:                    
            cntLst = [np.sum(noteGroupMap[y0:y1,x1:x1+(x1-x0)]==i) for i in ngIdxLst]
            ngIdx = ngIdxLst[cntLst.index(max(cntLst))]
        if len(stemIdxLst)>1:   
            ngx0, _, ngx1, _ = noteGroupVerticallyMerged[ngIdx].boundingBox
            cntLst = [np.sum(np.sum(stemIdxMap[y0:y1,ngx0:ngx1]==i,0)>0) for i in stemIdxLst]
            stemIdx = stemIdxLst[cntLst.index(max(cntLst))]
        if stemIdx < len(noteGroupVerticallyMerged[ngIdx].noteStemList):
            noteGroupVerticallyMerged[ngIdx].noteStemList[stemIdx].accidentals = sfnc.getValue()
            sfnc.ngIndex = ngIdx
    for grps in sfnGroupList:
        currNgIdx = sfnClefList[grps[-1]].ngIndex
        if currNgIdx is None:
            sfnClefList[grps[-1]].endKeySignature = True 
            for grr in grps:
                sfnClefList[grr].isKeySignature = True
            continue
        currNg:NoteGroup = noteGroupVerticallyMerged[currNgIdx]
        if len(currNg.noteStemList)==1:
            continue
        ngx0, _, ngx1, _ = currNg.boundingBox
        for grr in grps[:-1]:
            currSfn = sfnClefList[grr]
            y0,y1 = currSfn.shrinkYs
            stemIdxLst = np.unique(stemIdxMap[y0:y1,ngx0:ngx1]).tolist()
            if -1 in stemIdxLst: 
                stemIdxLst.remove(-1)
            if len(stemIdxLst)>=1:                    
                cntLst = [np.sum(np.sum(stemIdxMap[y0:y1,ngx0:ngx1]==i,0)>0) for i in stemIdxLst]
                stemIdx = stemIdxLst[cntLst.index(max(cntLst))]
                if currNg.noteStemList[stemIdx].accidentals is None:
                    noteGroupVerticallyMerged[currNgIdx].noteStemList[stemIdx].accidentals = currSfn.getValue()
                    noteGroupVerticallyMerged[currNgIdx].noteStemList[stemIdx].accidentalBox = currSfn.boundingBox
                else:
                    print('Error assigning sfn to notes')


    accidentalsColors = [(255,255,0),(255,0,125),(255,0,255)] # flat, natural, sharp
    for ng in noteGroupVerticallyMerged:
        if ng is None:
            continue
        for idx, stem in enumerate(ng.noteStemList):
            if stem.accidentals is not None:
                x0,y0,x1,y1 = stem.noteBox
                currColor = accidentalsColors[stem.accidentals+1]
                accidentalsImg = cv2.rectangle(accidentalsImg, (x0,y0),(x1,y1),currColor,3, cv2.LINE_AA)
    imwrite("assignedAccidentals.jpg",accidentalsImg,strictlyYes=True)
    return stemIdxMap, noteGroupVerticallyMerged, {'assignedAccidentals':accidentalsImg}

def extendNotegroupsToStaff(noteGroupMap:np.ndarray, 
                            noteGroupVerticallyMerged:List[NoteGroup],
                            staffObjList:List[Staff],
                            beamMapImg:np.ndarray):
    bb,gg,rr = cv2.split(beamMapImg)
    ngZero = np.zeros_like(noteGroupMap)
    for sf in staffObjList:
        ngZero[sf.ys[0]:sf.ys[-1], sf.left:sf.right] = 1
    for idx, ng in enumerate(noteGroupVerticallyMerged):
        if ng is None:
            continue
        x0,y0,x1,y1 = ng.boundingBox
        if np.sum(ngZero[y0:y1,x0:x1])==0:
            clefNo = round(np.mean(rr[y0:y1,4]))
            if len(np.unique(noteGroupMap[y0:y1,x0:x1])) == 1:
                noteNo =np.unique(noteGroupMap[y0:y1,x0:x1])[0]
                if y0>staffObjList[clefNo-1].ys[2]: # is at the bottom
                    y00 = staffObjList[clefNo-1].ys[-2]
                    y11 = y1
                else:
                    y00 = y0
                    y11 = staffObjList[clefNo-1].ys[1]            
                gg[y00:y11,x0:x1] = 1
                noteGroupMap[y00:y11,x0:x1]=noteNo
                rr[y00:y11,x0:x1] = clefNo
                ng.boundingBox = (x0,y00,x1,y11)

    return cv2.merge([bb,gg,rr]), noteGroupMap

def constructBar(noteGroupMap:np.ndarray, 
                   stemIdxMap: np.ndarray,
                   noteGroupVerticallyMerged: List[NoteGroup|None],
                   restMap: np.ndarray,
                   restList:List[Rest|None],
                   sfnClefMap:np.ndarray,
                   sfnClefList:List[Union[Accidentals,Clef, None]],
                   beamMapImg:np.ndarray,
                   staffList:List[Staff]):
    numLines = len(staffList)
    bb,gg,rr = cv2.split(beamMapImg)
    barList:List[List[Bar]] = [[] for _ in range(numLines)]
    numBarsEachLine:List[int] = [0 for _ in range(numLines)]
    mapForMatching:List[np.ndarray|None] = [None,noteGroupMap,sfnClefMap, sfnClefMap, None, restMap]
    listForMatching:List[List] = [None, noteGroupVerticallyMerged,sfnClefList, sfnClefList, None, restList]
    notFinishedBar = None
    allRanges = []
    for kk in range(numLines):
        allBars = []
        sf0 = staffList[kk]
        barPlace = np.unique(np.where(gg[sf0.ys[0]:sf0.ys[-1], sf0.left:]==4)[1])
        isBar = np.sort(barPlace)
        ranges = []
        points = [0,isBar[0]] 
        i = 1
        while i < len(isBar):
            while isBar[i] == isBar[i-1]+1 and i<len(isBar)-1:
                i+=1
            if i == len(isBar)-1 and isBar[i] == isBar[i-1]+1:
                break
            points.append(isBar[i])
            i+=1
        for i in range(len(points)-1):
            if points[i+1]>points[i]+sf0.get_yOne():
                ranges.append((points[i]+sf0.left, points[i+1]+sf0.left))
        allRanges.append(ranges)
    for t in range(numLines): # trackNo
        printedLineNo = t # the 
        currBarList:List[Bar] = []
        sf0 = staffList[printedLineNo]
        ranges = allRanges[t]
        for ridx, rng in enumerate(ranges):
            if notFinishedBar is not None:
                bar = notFinishedBar
                notFinishedBar = None
            else:
                bar = Bar()
            currX = rng[0]
            while currX<rng[1]:
                lineLst = np.unique(gg[sf0.ys[0]-sf0.get_yOne():sf0.ys[-1]+sf0.get_yOne(), currX]).tolist()
                if 0 in lineLst:
                    lineLst.remove(0)
                if len(lineLst)>1:
                    for i in [1,5,3,2]:
                        if i in lineLst:
                            lineLst = [i]
                            break
                if len(lineLst)==0 or 4 in lineLst:
                    currX+=1
                else:
                    typeId = lineLst[0]
                    currMap = mapForMatching[typeId]
                    currLst = listForMatching[typeId]
                    inMapId = np.unique(currMap[sf0.ys[0]-sf0.get_yOne():sf0.ys[-1]+sf0.get_yOne(), currX]).tolist()
                    if 0 in inMapId:
                        inMapId.remove(0)
                    if len(inMapId)==1: 
                        sanityCheck = True
                        if typeId == 3: # clef
                            currClef:Clef = sfnClefList[inMapId[0]]
                            _,yy0,_,yy1 = currClef.boundingBox
                            if yy1-yy0 < sf0.get_yOne_float()*3:
                                sanityCheck = False
                            if currClef.type ==0:
                                if yy0<sf0.ys[0]-sf0.get_yOne()/3:
                                    currClef.type = -2
                        if not sanityCheck:
                            # don't add the element if it's suspicious
                            currX = currLst[inMapId[0]].boundingBox[2]+1
                        elif typeId == 1:
                            x0,y0,x1,y1 = currLst[inMapId[0]].boundingBox
                            uniq = np.unique(rr[y0:y1, x0:x1]).tolist()
                            if 0 in uniq:
                                uniq.remove(0)
                            if len(uniq) != 1:
                                currX+=1
                            elif uniq[0] != printedLineNo+1:
                                currX = currLst[inMapId[0]].boundingBox[2]+1
                            else:
                                bar.addElement(currLst[inMapId[0]])
                                if (currX == currLst[inMapId[0]].boundingBox[2]+1):
                                    print()
                                currX = currLst[inMapId[0]].boundingBox[2]+1
                        else:
                            bar.addElement(currLst[inMapId[0]])
                            currX = currLst[inMapId[0]].boundingBox[2]+1
                    else:
                        currX+=1
                        print("has more than 1 id")
            currBarList.append(bar)
        
        nextIdx = np.max(np.unique(np.where(gg[sf0.ys[0]-sf0.get_yOne():sf0.ys[-1]+sf0.get_yOne(), rng[1]:]>0)[1]))
        if nextIdx>5:
            rng = (isBar[-1]+sf0.left+1, rng[1]+nextIdx)
            bar = Bar()
            currX = rng[0]
            while currX<rng[1]:
                lineLst = np.unique(gg[sf0.ys[0]-sf0.get_yOne():sf0.ys[-1]+sf0.get_yOne(), currX]).tolist()
                if 0 in lineLst:
                    lineLst.remove(0)
                if len(lineLst)>1:
                    for i in [1,5,3,2]:
                        if i in lineLst:
                            lineLst = [i]
                            break
                if len(lineLst)==0 or 4 in lineLst:
                    currX+=1
                else:
                    typeId = lineLst[0]
                    currMap = mapForMatching[typeId]
                    currLst = listForMatching[typeId]
                    inMapId = np.unique(currMap[sf0.ys[0]-sf0.get_yOne():sf0.ys[-1]+sf0.get_yOne(), currX]).tolist()
                    if 0 in inMapId:
                        inMapId.remove(0)
                    if len(inMapId)==1:
                        bar.addElement(currLst[inMapId[0]])
                        currX = currLst[inMapId[0]].boundingBox[2]+1
                    else:
                        currX+=1
                        print("has more than 1 id")
            notFinishedBar = bar
        barList[t] = currBarList
        numBarsEachLine[t] = len(currBarList)
        # in order of vln1's 1st line, 2nd line ... | vln2's 1st line, 2nd line ...
    return barList,allRanges, numBarsEachLine



def find_best_rotation(staffObjList):
    def compute_line_angle(y_values, x_start, x_end):
        n = len(y_values)
        x = np.linspace(x_start, x_end, n)
        
        # Fit line y = mx + b
        m, _ = np.polyfit(x, y_values, 1)
        
        angle = np.arctan(m)  # radians
        return angle
    angles = []
    
    for oneStaff in staffObjList:
        angle = compute_line_angle(
            oneStaff.yUpperList,
            oneStaff.left,
            oneStaff.right
        )
        angles.append(angle)
        angle = compute_line_angle(
            oneStaff.yLowerList,
            oneStaff.left,
            oneStaff.right
        )
        angles.append(angle)
    best_angle = np.median(angles)
    return np.degrees(best_angle)

def clearExtraMapping(staffList: List[Staff], dataDict, numInstrument: int):
    insToDelete = len(staffList)%numInstrument
    medStaff = statistics.median([sf.get_yOne_float() for sf in staffList])
    toMedDiff = [abs(sf.get_yOne_float()-medStaff) for sf in staffList]
    indices = sorted(range(len(toMedDiff)), key=lambda i: toMedDiff[i], reverse=True)[:insToDelete]
    for i in sorted(indices, reverse=True):
        del staffList[i]
    rangeInPair = [0]
    for sf0 in staffList:
        rangeInPair.append(sf0.ys[0]-int(sf0.get_yOne_float()*5))
        rangeInPair.append(sf0.ys[-1]+int(sf0.get_yOne_float()*5))
    pageheight = dataDict['image'].shape[0]
    rangeInPair.append(pageheight)
    pairs = list(zip(rangeInPair[::2], rangeInPair[1::2]))
    for ddKey in dataDict:
        for (st,ed) in pairs:
            dataDict[ddKey][st:ed,:] = dataDict[ddKey][0,0]
    return dataDict

# ==================================================================================================
# main function
# ==================================================================================================
def png2decode(img_name: str, img_path: str, instrumentNumTrack: int = 0, packedImg:bool = False):
    base_path = img_path.replace('.png','')
    npy_path = f"{base_path}.npy"
    pkl_path = f"{base_path}_staffList.pkl"
    barList_path = f"{base_path}_barlist.pkl"

    # using oemer's output
    if not os.path.exists(npy_path):
        printt(f"missing npy for {img_name}")
        dataDictOrigin = runModel1(img_path,outputPath = base_path,dodewarp=False,save_npy=True)
    if not os.path.exists(pkl_path):
        printt(f"missing pkl for {img_name}")
        staffListOrigin = runModel2(npy_path, img_path, base_path, packedImg)
    if os.path.exists(npy_path) and os.path.exists(pkl_path):
        printt(f"working on {img_path}")
    dataDict = np.load(npy_path,allow_pickle=True)
    dataDict = dataDict.tolist()
    
    # make the image bigger to ensure distance between two stafflines are > 8px
    for k in dataDict.keys():
        img = dataDict[k]
        resized_image = cv2.resize(img, None, fx=ORIGINAL_IMAGE_RESIZE_RATIO, fy=ORIGINAL_IMAGE_RESIZE_RATIO, interpolation=cv2.INTER_NEAREST)
        dataDict[k] = resized_image

    outputDebugStemRestImg(dataDict) # generate stemrests_ in debug
    outputDebugSymbolsImg(dataDict) # generate symbols_ in debug

    bar_height, staffObjList = init_bar_height(dataDict, min_barheight=MIN_BAR_HEIGHT)

    # !!! 五線譜
    # staff.left & staff.right: 五線譜開始/結束的位置
    # staff.ys: [y0, y1, y2, y3, y4] 五線譜每行的中心位置
    # !!! add csv decoding for instruments here
    print(f'barheight: {bar_height}')
    print(bar_height)
    bestRotation = find_best_rotation(staffObjList)
    if abs(bestRotation) > 0.05:
        for k in dataDict.keys():
            img = dataDict[k]
            fillVal = 0
            if k == 'image':
                fillVal = 255
            resized_image = rotate(img, angle=bestRotation, reshape=True, mode='constant', cval=fillVal)
            dataDict[k] = resized_image

        bar_height, staffObjList = init_bar_height(dataDict, min_barheight=MIN_BAR_HEIGHT)
    stepSize = int(bar_height/4)
    if instrumentNumTrack != 0:
        dataDict=clearExtraMapping(staffObjList, dataDict, instrumentNumTrack)

    beamNoClefKeyWithStemRest, beamWithoutStemRest, beam_nostem = getBeamImage(dataDict, bar_height, staffObjList)
    
    # get the gradient beamMap and (dont filter noteheadInitial with longer beams cause might remove noteheads)
    beamMapImg, debugImages = generateBeamStaffImg(beam_nostem, staffObjList, bar_height,stepSize=stepSize)

    noteheadBoxesInitList, noteBwImageOrigin = getInitialNoteheadBoxList(dataDict, beam_nostem, beamMapImg, bar_height)

    # mask the note image with the notehead boxes
    maskedNoteBwImage = maskImage(noteheadBoxesInitList, noteBwImageOrigin)

    stem_init_list, image, debugImages= getStemList(dataDict, bar_height, noteheadBoxesInitList, beamNoClefKeyWithStemRest)
    writeDebugImagesFromDict(debugImages)
    
    stem_list_assigned_height, debugImages = assignStemLength(stem_init_list, image, beamMapImg, stepSize, bar_height)
    writeDebugImagesFromDict(debugImages)

    beamRemoveNotes = removeNotes(maskedNoteBwImage, beamNoClefKeyWithStemRest, beamMapImg)
    imwrite(f'knn_beam.jpg', beamRemoveNotes)
    
    beam_img2, stem_list_assigned, beam_heights = assignBeamLengthBeamImg(stem_list_assigned_height, image, beamRemoveNotes, bar_height)
    imwrite(f'knn_beamImg.jpg', beam_img2)
    
    noteGroupList,  beamMapImg, noteGroupStemMap, noteGroupMap, debugImages = knnRhythmAndDraw(stem_list_assigned, beam_heights, image, bar_height, beamMapImg, stemUpClassifier=STEM_UP_MODEL, stemDownClassifier=STEM_DOWN_MODEL,img_name=img_name)
    outputImWrite(f"{img_name}_knnbeams2.jpg", debugImages['knnbeams2'])
    writeDebugImagesFromDict(debugImages)
    outputNotegroupStemMapImg(noteGroupStemMap, image)
    outputNoteGroupMapImg(noteGroupMap, image)
    outputNoteMapImg(beamMapImg, image, staffObjList)

    restMap, restList, beamMapImg, debugImages = findRests(dataDict, beamMapImg, bar_height, REST_CLASSIFIER)

    noteGroupVerticallyMerged, noteGroupMapWithIssues, beamMapImg, debugImages = mergeVerticalNoteGroups(staffObjList,noteGroupList, noteGroupMap, beamMapImg, image,restMap, restList)
    writeDebugImagesFromDict(debugImages)

    sfnClefList, sfnClefMap, debugImages = symbol_classification(dataDict, SFN_CLEF_CLASSIFIER, bar_height)
    writeDebugImagesFromDict(debugImages)

    exportYolo(sfnClefList, image,img_name, bar_height)

    sfnClefList, sfnClefMap, beamMapImg = filterSfnClefModifyBeamMap(sfnClefList, noteGroupMap, beamMapImg)
    outputSfnClefNoteWhiteImg(beamMapImg, image, bar_height, noteGroupMap, staffObjList)
    outputSfnClefNoteImg(image, beamMapImg)

    staffEnhancedImg = enhanceStaff(image, staffObjList,bar_height)

    noteGroupPitchList, debugImages = assignPitch(staffEnhancedImg,noteGroupList, beamMapImg, bar_height)
    writeDebugImagesFromDict(debugImages)
    outputImWrite(f'{img_name}_pitchline.jpg', debugImages['pitchline'])

    noteChunkList, horizontalGroupImg = getNoteChunks(image, noteGroupMap, beamMapImg, noteGroupVerticallyMerged)
    outputImWrite(f'{img_name}_horizontalGroupImg.jpg',horizontalGroupImg)
    
    beamMapImg, barlineboxes, barlineSingle = findBarlines(image, beamMapImg, staffObjList, noteGroupStemMap, thres=0.8) # VARIABLE thres
    outputImWrite(f'{img_name}_barline.jpg',barlineSingle)
    outputSfnClefNoteBarlineRestImg(image, beamMapImg)
    dotPrevious = outputThingsBeforeDots(image, beamMapImg)
    outputImWrite(f'{img_name}_DotPrevious.jpg', dotPrevious)

    noteGroupVerticallyMerged, debugImages = assignDots(noteGroupVerticallyMerged,beamMapImg, image,bar_height,staffObjList)
    writeDebugImagesFromDict(debugImages)

    restList, debugImages = assignRestDots(restList, beamMapImg, noteGroupStemMap, image, bar_height, staffObjList)
    writeDebugImagesFromDict(debugImages)

    stemIdxMap, noteGroupVerticallyMerged, debugImages = assignSfnToNote(image, noteGroupMap, noteGroupVerticallyMerged,sfnClefMap,sfnClefList)
    writeDebugImagesFromDict(debugImages)

    beamMapImg,noteGroupMap = extendNotegroupsToStaff(noteGroupMap, noteGroupVerticallyMerged,staffObjList,beamMapImg)

    return noteGroupMap, stemIdxMap, noteGroupVerticallyMerged, restMap,restList,sfnClefMap,sfnClefList,beamMapImg,staffObjList, dataDict


if __name__ == '__main__':
    noteGroupMap: np.ndarray
    stemIdxMap: np.ndarray
    noteGroupVerticallyMerged: List[NoteGroup | None]
    restMap: np.ndarray
    restList: List[Rest | None]
    sfnClefMap: np.ndarray
    sfnClefList: List[Union[Accidentals, Clef, None]]
    beamMapImg: np.ndarray
    staffList: List[Staff]

    (
        noteGroupMap,
        stemIdxMap,
        noteGroupVerticallyMerged,
        restMap,
        restList,
        sfnClefMap,
        sfnClefList,
        beamMapImg,
        staffList,
        dataDict
    ) = png2decode("tchai_4_001", r"orch_dataset\tchai_4\images\001\tchai_4_001.png")

# 五線譜位置 ()