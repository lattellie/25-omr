

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from utils import *
from png2decode import png2decode
from pdf2png import savePdf2Png
import glob

if __name__ == '__main__':

    pieceName = 'dvo'
    if not os.path.exists(f'data_dataset/{pieceName}/imgs/{pieceName}_1/{pieceName}_1.png'):
        savePdf2Png(f'data_dataset/{pieceName}', 1, False, dpi=200)
    allImgPath = glob.glob(f'data_dataset/{pieceName}/imgs/{pieceName}*/{pieceName}*.png')

    for idx in range(len(allImgPath)):
        if os.path.exists(f'data_dataset/{pieceName}/debug/debug_{idx+1}.txt'):
            print(f"skipping page {idx+1}")
            continue
        imgPath = f'data_dataset/{pieceName}/imgs/{pieceName}_{idx+1}/{pieceName}_{idx+1}.png'
        noteGroupMap: np.ndarray
        stemIdxMap: np.ndarray
        noteGroupVerticallyMerged: List[NoteGroup | None]
        restMap: np.ndarray
        restList: List[Rest | None]
        sfnClefMap: np.ndarray
        sfnClefList: List[Union[Accidentals, Clef, None]]
        beamMapImg: np.ndarray
        staffList: List[Staff]
        imgName = os.path.basename(imgPath).split('.')[0]

        try:
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
            ) = png2decode(imgName, imgPath)
            

            # sfn_colors = [(255,255,0),(255,0,125),(255,0,255),(255,0,0),(0,255,0),(0,0,255)]
            # sfn_names = ['flat', 'natural', 'sharp','BassF', 'ViolaC', 'trebleG']
            clef_colors = [(0,255,0),(0,0,255),(255,0,0)]
            clef_yolo = [0,1,2] # 0: alto, 1: treble, 2: bass
            sfn_colors = [(255,0,125),(255,0,255),(255,255,0)]
            sfn_yolo = [3,4,5] # 3: natural, 4: sharp, 5: flat
            rest_colors = [(255,0,0),(0,0,255),(255,255,0),(0,120,255),(40,255,40)]
            rest_yolo = [6,7,8,9,10] # 6: 1/4, 7: 1/8, 8: 1/16, 9:1/32, 10: half or whole
            knn_colors = [(255,0,0),(0,0,255),(255,255,0),(0,120,255),(40,255,40), (245, 220, 255),(230,130,175),(165,170, 70)]
            knn_yolo = [11,12,13,14,15,16,17,18] # 11: 1/4, 12: 1/8, 13: 1/16, 14: 1/32, 15: 1/64, 16:1/128, 17: 1/2, 18: whole
            knn_stem = [19,20,21,22,23,24,25,26] # 19: 1/4's stem, 20: 1/8 ... etc.
            img = dataDict['image'].copy()
            labelList = []
            h, w = img.shape[:2]
            def add_yolo_label(class_id, x0, y0, x1, y1):
                x_center = ((x0 + x1) / 2) / w
                y_center = ((y0 + y1) / 2) / h
                bw = (x1 - x0) / w
                bh = (y1 - y0) / h
                labelList.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"
                )

            for rest in restList:
                if rest is not None:
                    x0, y0, x1, y1 = rest.boundingBox
                    img = cv2.rectangle(img, (x0,y0), (x1,y1), rest_colors[rest.rhythm], 3, cv2.LINE_AA)
                    add_yolo_label(rest_yolo[rest.rhythm], x0, y0, x1, y1)
            for ng in noteGroupVerticallyMerged:
                if ng is not None:
                    for stem in ng.noteStemList:
                        x0,y0,x1,y1 = stem.noteBox
                        add_yolo_label(knn_yolo[stem.rhythm], x0, y0, x1, y1)
                        img = cv2.rectangle(img, (x0,y0), (x1,y1), knn_colors[stem.rhythm], 2, cv2.LINE_AA)
                        add_yolo_label(knn_stem[stem.rhythm], stem.getX()-3, stem.getTopCoord()[1], stem.getX()+3, stem.getBottomCoord()[1])
                        img = cv2.rectangle(img, (stem.getX()-3, stem.getTopCoord()[1]), (stem.getX()+3, stem.getBottomCoord()[1]), knn_colors[stem.rhythm], 2, cv2.LINE_AA)
            for sfn in sfnClefList:
                if type(sfn) == Accidentals:
                    x0,y0,x1,y1 = sfn.getBbox()
                    add_yolo_label(sfn_yolo[sfn.getValue()], x0, y0, x1, y1)
                    img = cv2.rectangle(img, (x0,y0), (x1,y1), sfn_colors[sfn.getValue()], 1, cv2.LINE_AA)
                elif type(sfn) == Clef:
                    x0,y0,x1,y1 = sfn.getBbox()
                    add_yolo_label(clef_yolo[sfn.getValue()], x0, y0, x1, y1)
                    img = cv2.rectangle(img, (x0,y0), (x1,y1), clef_colors[sfn.getValue()], 1, cv2.LINE_AA)

            if not os.path.exists(f'data_dataset/{pieceName}/debug'):
                os.mkdir(f'data_dataset/{pieceName}/debug')
            cv2.imwrite(f'data_dataset/{pieceName}/debug/debug_{idx+1}.jpg',img)        
            with open(f'data_dataset/{pieceName}/debug/debug_{idx+1}.txt', "w") as f:
                f.write("\n".join(labelList))
            print()
        except:
            print(f"error processing, skipping {idx+1}")