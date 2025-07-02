import cv2
import os

from omr.transformer.configs import default_config
from omr.transformer.staff2score import Staff2Score
from omr.type_definitions import NDArray
from omr.logger import get_logger

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = get_logger(__name__)

inference: Staff2Score | None = None

tromr_width = 1280
tromr_height = 128
def reshape_to_tromr(ori_image: NDArray) -> NDArray:
    img_width = ori_image.shape[1]
    img_height = ori_image.shape[0]
    if img_width/img_height > tromr_width/tromr_height:
        # need to make image taller
        ori_image = cv2.resize(ori_image, (tromr_width, img_height*tromr_width//img_width))
        border = tromr_height-ori_image.shape[0]
        ori_image = cv2.copyMakeBorder(ori_image, border//2, border-border//2, 0,0, cv2.BORDER_CONSTANT, value=(255,255,255))
    else:
        ori_image = cv2.resize(ori_image, (img_width*tromr_height//img_height, tromr_height))
        border = tromr_width-ori_image.shape[1]
        ori_image = cv2.copyMakeBorder(ori_image, 0,0, border//2, border-border//2, cv2.BORDER_CONSTANT, value=(255,255,255))
    return ori_image

def predict_best(
    org_image: NDArray,
    image_path='',
    saveTxt=False
) -> str:
    global inference  # noqa: PLW0603
    if inference is None:
        inference = Staff2Score(default_config)

    new_image = reshape_to_tromr(org_image)
    print(new_image.shape)
    cv2.imwrite('temp.jpg', new_image)
    result = inference.predict(
        new_image
    )
    print(result)
    if len(result)==1:
        output_str = result[0]
        if saveTxt:
            with open(image_path.rsplit('.',1)[0]+'.txt', 'w') as file:
                file.write(output_str)
    else:
        logger.error(f'length of result: {len(result)}')
        return ''
    return output_str

if __name__ == '__main__':
    # run in tromr_main so that it has the right import
    doNotRepeat = True
    folder_path = 'C:/Ellie/ellie2023~2024/iis/omr-iis/images/staff_test'
    files = os.listdir(folder_path)
    files = [folder_path+'/'+file for file in files if file.endswith('.png')]
    for imgpath in files:
        outpath = imgpath.replace('.png','.txt')
        if os.path.exists(outpath) and not doNotRepeat:
            continue
        currImg = cv2.imread(imgpath)
        outstr = predict_best(currImg, image_path = imgpath)
        with open(outpath, 'w') as file:
            file.write(outstr)
        print(outstr)
