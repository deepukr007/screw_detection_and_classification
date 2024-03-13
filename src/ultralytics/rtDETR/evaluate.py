from ultralytics import RTDETR
import os
from time import time 
from glob import glob

if __name__ == "__main__":
    
    model_path = "/home/krishnar@iff.intern/thesis/screw_detection/src/ultralytics/rtDETR/train_screw_detr/epoch_100_batch_20_imgz_640"


# Todo chnage
    data_conf = {
        "crs" : '/home/krishnar@iff.intern/thesis/screw_detection/yolov8/dataset_conf/screw_data.yaml', 
        "weee" : '/home/krishnar@iff.intern/thesis/screw_detection/yolov8/dataset_conf/WEEE_data.yaml',
        "iff": '/home/krishnar@iff.intern/thesis/screw_detection/yolov8/dataset_conf/iff_screw_data.yaml'
    }

    for model in glob((os.path.join(model_path , "**/best.pt")), recursive=True):
            weight= model.split('/')[-1]
            model = RTDETR(model)
            start = time()
            results = model.val(data=data_conf["crs"]  , split="val" , workers=8 ,save_json=True ,batch=1 , imgsz=640 ,  device="0" , plots=True , conf=0.3, project="640" , name="val")
            total_time = time() - start 
            print("Total Time for evaluation : " , total_time , flush=True)
