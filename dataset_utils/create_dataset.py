import os
from glob import glob
import cv2 as cv
from pathlib import Path
from ultralytics.data.converter import convert_coco


path_to_images = "/home/data/iDear/Own2_processed"

datasets_common_path = "/home/krishnar@iff.intern/thesis/datasets"

datset_name = 'own2' 

selection_stratergy = "mid"

coco_label_path ="/home/krishnar@iff.intern/thesis/screw_detection_and_classification/dataset_utils"



def select_image(m_img , stratergy):
    if stratergy == "mid":
        return m_img [2]


datset_path = Path(f"{datasets_common_path}/{datset_name}")
im_path = os.path.join(datset_path  , "images")
label_path = os.path.join(datset_path  , "labels")

if os.path.exists(datset_path) is False:
    os.makedirs(im_path) 
    os.makedirs(label_path , exist_ok=True ) 

count = 0 
for i in glob(os.path.join(path_to_images , "*.tif")):
    img_name = Path(i).stem
    ret , m_img = cv.imreadmulti(i)
    img = select_image(m_img , selection_stratergy)
    img_file = Path(f"{im_path}/{img_name}.jpg")
    cv.imwrite(str(img_file), img)
    print(f"{img_name} is created")


convert_coco(coco_label_path, save_dir = label_path ,  use_segments=False, use_keypoints=False)