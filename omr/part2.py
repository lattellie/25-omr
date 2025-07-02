import numpy as np
import cv2
from omr import layers
from pathlib import Path
import statistics
from typing import Tuple, List
import pickle
import os
from omr.staffline_extraction import staff_extract
from omr.notehead_extraction import note_extract
from omr.symbol_extraction import symbol_extract
from omr.note_group_extraction import group_extract
from omr.logger import get_logger
logger = get_logger(__name__)
import warnings
warnings.filterwarnings("ignore")
# the "staff" object we are going to parse in tromr, need info about:
# 1. where 
class OmrStaff:
    class StaffElements:
        # maybe refactor typ to enum type instead of strings
        def __init__(self, x_left: int, bbox: Tuple[int,int,int,int], typ: str, addval: int):
            self.x_left = x_left
            self.bbox = bbox
            self.typ = typ
            # typ: one of clef, note, rest, sfn, barline
            self.addval = addval
            self.newLabel: str|None = None
            self.flag: int|None = 0
        def addLabel(self, label:str):
            self.newLabel = label
        def setFlag(self, flag:int):
            self.flag = flag
        def display_element(self):
            if self.newLabel is None:
                print(f"typ: {self.typ[:3]}, x: {str(self.x_left).rjust(4, ' ')}, addval: {self.addval}, flag: {self.flag}")
            else:
                print(f"typ: {self.typ[:3]}, x: {str(self.x_left).rjust(4, ' ')}, addval: {self.addval}, flag: {self.flag}, newLabel: {self.newLabel}")

    def __init__(self) -> None:
        self.top_left: Tuple[int,int] = None
        self.bottom_right: Tuple[int,int] = None
        # self.staff: self.Staff = None
        self.staff_left: int = None
        self.staff_right: int = None
        self.ys: List[int] = None
        self.elements: List[self.StaffElements] = []

    def set_staff(self, left:int, right:int, ys:List[int]):
        self.staff_left: int = left
        self.staff_right: int = right
        self.ys: List[int] = ys
        self.top_left = (left, ys[0])
        self.bottom_right = (right, ys[-1])
    
    def add_element(self, x_left: int, bbox: Tuple[int,int,int,int], typ: str, addval: int):
        elem = self.StaffElements(x_left, bbox, typ, addval)
        self.elements.append(elem)
    
    def sort_element(self):
        self.elements = sorted(self.elements, key = lambda obj: obj.x_left)
    
    def print_staff(self):
        print(f"Printing Staff with y:{self.ys[0]}~{self.ys[-1]}")
        for i in range(len(self.elements)):
            elm = self.elements[i]
            # print(f'{elm.typ[:3]}: x={str(elm.x_left).rjust(4, ' ')}, addval={elm.addval}')
            # print(f"typ: {elm.typ[:3]}, x: {str(elm.x_left).rjust(4, ' ')}, addval: {elm.addval}.")
            print(f"typ: {elm.typ[:3]}, x: {str(elm.x_left).rjust(4, ' ')}, addval: {str(elm.addval).rjust(2, ' ')}, flag: {elm.flag}, newLabel: {elm.newLabel}")
    
    def print_staff_G2(self):
        print(f"Printing Soprano Staff with y:{self.ys[0]}~{self.ys[-1]}")
        noteVect = ['D','E','F','G','A','B','C']
        for i in range(len(self.elements)):
            elm:OmrStaff.StaffElements = self.elements[i]
            if elm.typ=='note':
                note_disp = noteVect[elm.addval%len(noteVect)]
                print(f"typ: {elm.typ[:3]}, x: {str(elm.x_left).rjust(4, ' ')}, addval: {str(note_disp).rjust(2, ' ')}, flag: {elm.flag}, newLabel: {elm.newLabel}")
            else:
                print(f"typ: {elm.typ[:3]}, x: {str(elm.x_left).rjust(4, ' ')}, addval: {str(elm.addval).rjust(2, ' ')}, flag: {elm.flag}, newLabel: {elm.newLabel}")

    
    def get_bbox(self,typ:None|str=None):
        if typ is None:
            return [self.top_left[0], self.top_left[1], self.bottom_right[0], self.bottom_right[1]]
        if typ[0] =='l':
            return self.top_left[0]
        elif typ[0]=='r':
            return self.bottom_right[0]
        elif typ[0]=='t':
            return self.top_left[1]
        elif typ[0]=='b':
            return self.bottom_right[1]
        else:
            return [self.top_left[0], self.top_left[1], self.bottom_right[0], self.bottom_right[1]]
    
    def get_yOne(self):
        x = [self.ys[i+1] - self.ys[i] for i in range(len(self.ys) - 1)]
        return sum(x)//len(x)
    
    def set_bbox(self, left:None|int=None, right:None|int=None, top:None|int=None, bottom:None|int=None):
        if left:
            self.top_left = (left, self.top_left[1])
        if right:
            self.bottom_right = (right, self.bottom_right[1])
        if top:
            self.top_left = (self.top_left[0],top)
        if bottom:
            self.bottom_right = (self.bottom_right[0], bottom)
    
    def expand_bound(self):
        self.top_left = (int(min(min([min(elm.bbox[0], elm.bbox[2]) for elm in self.elements]), self.top_left[0])),
                         int(min(min([min(elm.bbox[1], elm.bbox[3]) for elm in self.elements]), self.top_left[1])))
        self.bottom_right = (int(max(max([max(elm.bbox[0], elm.bbox[2]) for elm in self.elements]), self.bottom_right[0])),
                             int(max(max([max(elm.bbox[1], elm.bbox[3]) for elm in self.elements]), self.bottom_right[1])))
        print(f"expanding bounds to {self.top_left}, {self.bottom_right}")
    
    # typ: one of clef, note, rest, sfn, barline
    def drawRect(self, image, savedir, saveImg = True, col = (0,0,255)):
        img = image.copy()
        img = cv2.rectangle(img, self.top_left, self.bottom_right, col, 2)
        for i in range(len(self.elements)):
            e = self.elements[i]
            img = cv2.rectangle(img, (e.bbox[0], e.bbox[1]), (e.bbox[2], e.bbox[3]), col, 2)
        if saveImg:
            cv2.imwrite(savedir+'_rect.jpg', img)
        return img
    
def staffs_to_omrStaffList(staffs):
    left = min([sf.x_left for sf in staffs[0,:]])
    right = max([sf.x_right for sf in staffs[-1,:]])
    omrstaff_list: List[OmrStaff] = []
    staffRange = []
    for i in range(staffs.shape[1]):
        top = statistics.median([sf.y_upper for sf in staffs[:,i]])
        bottom = statistics.median([sf.y_lower for sf in staffs[:,i]])
        unit_size = statistics.median([sf.unit_size for sf in staffs[:,i]])
        ys = [int(top+unit_size*n) for n in range(5)]
        sf = OmrStaff()
        sf.set_staff(left, right, ys)
        omrstaff_list.append(sf)
        staffRange.append(range(int(top),int(bottom)))
    return omrstaff_list, staffRange

def filter_valid_notes(notes, lowerthres = 0.8, upperthres = 1.2):
    allsize = [abs((n.bbox[3]-n.bbox[1])*(n.bbox[2]-n.bbox[0])) for n in notes]
    allwidth = [abs((n.bbox[3]-n.bbox[1])) for n in notes]
    allheight = [abs((n.bbox[2]-n.bbox[0])) for n in notes]
    validNotes = (allsize>np.median(allsize)*lowerthres**2) & (allsize<np.median(allsize)*upperthres**2)
    validNotes = validNotes & (allwidth>np.median(allwidth)*lowerthres) & (allwidth<np.median(allwidth)*upperthres)
    validNotes = validNotes & (allheight>np.median(allheight)*lowerthres) & (allheight<np.median(allheight)*upperthres)
    return [note for note,flag in zip(notes,validNotes) if flag]


def saveNoteGroupImgs(notes, image, output_dir,name, saveImg=True):
    if saveImg:
        if not os.path.isdir(output_dir+'_debug'):
            os.mkdir(output_dir+'_debug')
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (125, 125, 0),  # 
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (255, 192, 192),# 
            (128, 255, 128),# 
            (128, 0, 128),  # Purple
            (255, 165, 0)   # Orange
        ]
        img = image.copy()
        for i in range(len(notes)):
            n = notes[i]
            color = colors[n.group%len(colors)]
            img = cv2.rectangle(img, (n.bbox[0], n.bbox[1]), (n.bbox[2],n.bbox[3]), color, 2)
        cv2.imwrite(output_dir+'_debug/'+name+'.jpg',img)

def reassignGroupAndNotes(groups, notes, staffRange):
    for i in range(len(groups)):
        g = groups[i]
        for j in range(len(staffRange)):
            if g.bbox[3] in staffRange[j] or g.bbox[1] in staffRange[j]:
                if (j==len(staffRange)-1 or (g.bbox[3] < staffRange[j+1][0])) and (j==0 or (g.bbox[1] > staffRange[j-1][-1])):
                    g.group = j
                    for k in range(len(g.note_ids)):
                        id = g.note_ids[k]
                        notes[id].group = j
                    continue
                else:
                    for k in range(len(g.note_ids)):
                        id = g.note_ids[k]
                        print(f"note: {id}, group: {notes[id].group}")
                    continue
                    
    return groups, notes

def filterBarlines(barlines,staffRange):
    selected = []
    for i in range(len(barlines)):
        b = barlines[i]
        st = staffRange[b.group][0]
        ed = staffRange[b.group][-1]
        expandedRange = range(st-(ed-st)//10, ed+(ed-st)//10)
        if b.bbox[1] in expandedRange and b.bbox[3] in expandedRange:
            selected.append(i)
    newbarlines = [barlines[i] for i in selected]
    return newbarlines

def filterRests(rests, staffRange):
    selected = []
    for i in range(len(rests)):
        b = rests[i]
        currRange = staffRange[b.group]
        st = staffRange[b.group][0]
        ed = staffRange[b.group][-1]
        oneLineHeight = len(currRange)//5
        if (b.bbox[1] in currRange or b.bbox[3] in currRange) and (abs(b.bbox[3]-b.bbox[1])>oneLineHeight*2):
            selected.append(i)
    newrests = [rests[i] for i in selected]
    return newrests

def filterSfns(sfns, staffRange):
    medWidth = np.median([abs((n.bbox[2]-n.bbox[0])) for n in sfns])
    selected = []
    for i in range(len(sfns)):
        sfn = sfns[i]
        oneLineHeight = len(staffRange[sfn.group])//5
        if abs(sfn.bbox[3]-sfn.bbox[1])*0.8>abs(sfn.bbox[2]-sfn.bbox[0]) and abs(sfn.bbox[3]-sfn.bbox[1])>2*oneLineHeight and abs(sfn.bbox[2]-sfn.bbox[0])>medWidth*0.5:
            selected.append(i)
    newsfns = [sfns[i] for i in selected]
    return newsfns

def addElement(elements, staffList, typ):
    # typ: one of clef, note, rest, sfn, barline
    if typ=='barline':
        for i in range(len(elements)):
            elem = elements[i]
            grp = elem.group
            staffList[grp].add_element(min(elem.bbox[0], elem.bbox[2]), elem.bbox, typ, 0)
            #  x_left: int, bbox: Tuple[int,int,int,int], typ: str, addval: int
    elif typ == 'note':
        for i in range(len(elements)):
            elem = elements[i]
            grp = elem.group
            staffList[grp].add_element(min(elem.bbox[0], elem.bbox[2]), elem.bbox, typ, elem.staff_line_pos)
    else:
        for i in range(len(elements)):
            elem = elements[i]
            grp = elem.group
            staffList[grp].add_element(min(elem.bbox[0], elem.bbox[2]), elem.bbox, typ, elem._label.value)
    return staffList

        


# helper function from oemer
def register_note_id() -> None:
    symbols = layers.get_layer('symbols_pred')
    layer = layers.get_layer('note_id')
    notes = layers.get_layer('notes')
    for idx, note in enumerate(notes):
        x1, y1, x2, y2 = note.bbox
        yi, xi = np.where(symbols[y1:y2, x1:x2]>0)
        yi += y1
        xi += x1
        layer[yi, xi] = idx
        notes[idx].id = idx

def loadModel1(img_path:str,model1Data:dict) -> str:
    img_path = Path(img_path)
    f_name = os.path.splitext(img_path.name)[0]
    stems_rests = model1Data['stems_rests']
    clefs_keys = model1Data['clefs_keys']
    notehead = model1Data['notehead']
    symbols = model1Data['symbols']
    staff = model1Data['staff']
    image = model1Data['image']

    layers.register_layer("stems_rests_pred", stems_rests)
    layers.register_layer("clefs_keys_pred", clefs_keys)
    layers.register_layer("notehead_pred", notehead)
    layers.register_layer("symbols_pred", symbols)
    layers.register_layer("staff_pred", staff)
    layers.register_layer("original_image", image)
    logger.info('finish loading all model1\'s data')
    return image

def extract(image, output_dir):
    # ---- Extract staff lines and group informations ---- #
    logger.info("Extracting stafflines")
    staffs, zones = staff_extract()
    layers.register_layer("staffs", staffs)  # Array of 'Staff' instances
    layers.register_layer("zones", zones)  # Range of each zones, array of 'range' object.
    
    # --- Get the info I need (a list of "big staff" object instead of chunks)
    staffList, staffRange = staffs_to_omrStaffList(staffs)
    logger.info("finish creating OmrStaff lists")

    # ---- Extract noteheads ---- #
    logger.info("Extracting noteheads")
    ori_notes = note_extract()
    saveNoteGroupImgs(ori_notes,image,output_dir,'notes_ori')
    # notes = filter_valid_notes(ori_notes)
    # saveNoteGroupImgs(notes,image,output_dir,'notes_valid')
    # skip the filtering for now
    notes = ori_notes

    symbols = layers.get_layer('symbols_pred')
    # Array of 'NoteHead' instances.
    layers.register_layer('notes', np.array(notes))
    # Add a new layer (w * h), indicating note id of each pixel.
    layers.register_layer('note_id', np.zeros(symbols.shape, dtype=np.int64)-1)
    register_note_id()

    # ---- Extract groups of note ---- #
    logger.info("Grouping noteheads")
    logger.info('ignornig all the note group reassigning in note_group_extraction: 329, part2: 260')
    groups, group_map = group_extract()
    saveNoteGroupImgs(groups,image,output_dir,'groups_ori')

    layers.register_layer('note_groups', np.array(groups))
    layers.register_layer('group_map', group_map)
    saveNoteGroupImgs(notes,image,output_dir,'notes_beforeReassign')

    groups, notes = reassignGroupAndNotes(groups, notes, staffRange)
    saveNoteGroupImgs(groups,image,output_dir,'groups_new')
    saveNoteGroupImgs(notes,image,output_dir,'notes_new')

    # ---- Extract symbols ---- #
    logger.info("Extracting symbols")
    barlines, clefs, sfns, rests = symbol_extract()

    
    # saveNoteGroupImgs(barlines, image,output_dir,'barline_ori')
    # barlines = filterBarlines(barlines,staffRange)
    # saveNoteGroupImgs(barlines, image,output_dir,'barline_new')
    # saveNoteGroupImgs(rests, image,output_dir,'rests_ori')
    # rests = filterRests(rests, staffRange)
    # saveNoteGroupImgs(rests, image,output_dir,'rests_new')
    # saveNoteGroupImgs(sfns, image, output_dir, 'sfns_ori')
    # sfns = filterSfns(sfns, staffRange)
    # saveNoteGroupImgs(sfns, image, output_dir, 'sfns_new')
    # addElement(clefs, staffList, 'clef')
    # addElement(notes, staffList, 'note')
    # addElement(rests, staffList, 'rest')
    # addElement(sfns, staffList, 'sfn')
    # addElement(barlines, staffList, 'barline')
    layers.register_layer('barlines', np.array(barlines))
    layers.register_layer('clefs', np.array(clefs))
    layers.register_layer('sfns', np.array(sfns))
    layers.register_layer('rests', np.array(rests))

    for i in range(len(staffList)):
        staffList[i].sort_element()
    
    # saving to the format I need
    with open(output_dir+'_staffList.pkl', "wb") as file:
        pickle.dump(staffList, file)
    cv2.imwrite(output_dir+'_image.jpg', image)

    return staffList

def runModel2(npy_path: str, img_path: str, output_dir: str):
    # output_dir will be images/tch/tch
    dataDict = np.load(npy_path,allow_pickle=True)
    dataDict = dataDict.tolist()
    image = loadModel1(img_path, dataDict)
    staffList = extract(image, output_dir)
    return staffList
