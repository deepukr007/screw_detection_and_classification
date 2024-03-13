from pprint import pprint
from ultralytics import YOLO
from utils import get_datset
import os
from time import time 
from glob import glob

if __name__ == "__main__":
    
    model_path = "/home/krishnar@iff.intern/thesis/screw_detection/src/ultralytics/yolov8/crs_yolo_adam/n_epochs_100_batch_16_imgsz1664/weights"
    dataset = "crs"

    imgz = 1664
    data = get_datset(dataset)
    split= "val"
    name  = f"{split}_{dataset}_{model_path.split('/')[-2]}"
   


    for model in glob((os.path.join(model_path , "**/best.pt")), recursive=True):
            model = YOLO(model)
            start = time()
            results = model.val(imgsz=imgz , data=data , split=split , workers=8 ,save_json=True ,project='results', device="0" , conf=0.1 , plots=True , name=name)
            total_time = time() - start 
            print("Total Time for evaluation : " , total_time , flush=True)
