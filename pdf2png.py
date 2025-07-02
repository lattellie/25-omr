from pdf2image import convert_from_path
import os
from PIL import Image

# yyy/xxx/ -> yyy/xxx/imgs/xxx_i/xxx_i.png
def savePdf2Png(root_folder:str, numPngPerPage:int, rotate:bool):
    file_name = os.path.basename(root_folder)
    pdf_path = os.path.join(root_folder, f'{file_name}.pdf')
    images = convert_from_path(pdf_path, dpi=300)
    imgNum = 1
    img_root = os.path.join(root_folder, 'imgs')
    if not os.path.isdir(img_root):
        os.mkdir(img_root)
    for fullImg in images:
        if rotate:
            fullImg = fullImg.transpose(Image.Transpose.ROTATE_270)
        if numPngPerPage == 1:
            image = fullImg
            folderPath = os.path.join(img_root, f'{file_name}_{imgNum}')
            if not os.path.isdir(folderPath):
                os.mkdir(folderPath)
            newroute = os.path.join(folderPath,f'{file_name}_{imgNum}.png')
            image.save(newroute, 'PNG')
            print(f"image {imgNum} saved to {newroute}")
            imgNum = imgNum+1
        else:
            for j in range(numPngPerPage):
                width, height = fullImg.size
                image = fullImg.crop((width//2*j, 0, (width//2)*(j+1), height))
                folderPath = os.path.join(img_root, f'{file_name}_{imgNum}')
                if not os.path.isdir(folderPath):
                    os.mkdir(folderPath)
                newroute = os.path.join(folderPath,f'{file_name}_{imgNum}.png')
                image.save(newroute, 'PNG')
                print(f"image {imgNum} saved to {newroute}")
                imgNum = imgNum+1
# if __name__ == '__main__':
#     savePdf2Png('string_dataset/pdf_data/beethoven1',1,False)


# # Path to the PDF file
# NUMPNG_PAGE = 1
# ROTATE = False
# pdf_path = input("enter pdf path [q to quit]: ")
# pdf_createFolder = input("create folder? [y/n]: ")
# creatFolder = True if pdf_createFolder.startswith("y") else False
# while not pdf_path.startswith("q"):
#     pdf_path = pdf_path.replace("\"","")
#     pdf_path = pdf_path.replace("\'","")

#     # Convert PDF to a list of images (one per page)
#     images = convert_from_path(pdf_path, dpi=390)

#     # Save each page as a PNG
#     imgNum = 1
#     for fullImg in images:
#         if ROTATE:
#             fullImg = fullImg.transpose(Image.Transpose.ROTATE_270)
#         if NUMPNG_PAGE == 1:
#             image = fullImg
#             if not creatFolder:
#                 newroute = pdf_path.replace(".pdf",f"_{imgNum}.png")
#                 image.save(newroute, 'PNG')
#                 print(f"image {imgNum} saved to {newroute}")
#             else:
#                 pdfName = os.path.basename(pdf_path)
#                 folderPath = pdf_path.replace(".pdf",f"_{imgNum}")
#                 if not os.path.isdir(folderPath):
#                     os.mkdir(folderPath)
#                 newroute = folderPath+"\\"+pdfName.replace(".pdf",f"_{imgNum}.png")
#                 image.save(newroute, 'PNG')
#                 print(f"image {imgNum} saved to {newroute}")
#             imgNum = imgNum+1

#         else:
#             for j in range(NUMPNG_PAGE):
#                 width, height = fullImg.size
#                 image = fullImg.crop((width//2*j, 0, (width//2)*(j+1), height))
#                 if not creatFolder:
#                     newroute = pdf_path.replace(".pdf",f"_{imgNum}.png")
#                     image.save(newroute, 'PNG')
#                     print(f"image {imgNum} saved to {newroute}")
#                 else:
#                     pdfName = os.path.basename(pdf_path)
#                     folderPath = pdf_path.replace(".pdf",f"_{imgNum}")
#                     if not os.path.isdir(folderPath):
#                         os.mkdir(folderPath)
#                     newroute = folderPath+"\\"+pdfName.replace(".pdf",f"_{imgNum}.png")
#                     image.save(newroute, 'PNG')
#                     print(f"image {imgNum} saved to {newroute}")
#                 imgNum = imgNum+1

#     pdf_path = input("enter pdf path: [q to quit]")

# print("Conversion complete!")
