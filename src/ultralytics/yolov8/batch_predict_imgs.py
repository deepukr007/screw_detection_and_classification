from ultralytics import YOLO
import cv2 
from glob import glob
import os
import torch
from time import time

torch.cuda.device(1)

if __name__ == "__main__":
    model_path = "/home/krishnar@iff.intern/thesis/screw_detection/src/ultralytics/yolov8/crs_yolo_sgd/n_epochs_100_batch_12_imgsz19202/weights/best.pt"
    #model_path = "/home/krishnar@iff.intern/thesis/screw_detection/src/ultralytics/yolov8/crs_yolo/n_epochs_100_batch_16_imgsz16643/weights/best.pt"
    result_path ="/home/krishnar@iff.intern/thesis/pred_res"
    dataset_path = "/home/krishnar@iff.intern/thesis/datasets/crs/validation/images"

    orig_lables = "/home/krishnar@iff.intern/thesis/datasets/crs/validation/labels"
   # dataset_path = "/home/krishnar@iff.intern/thesis/datasets/test"


    #file_path = os.path.dirname(os.path.abspath(__file__))
    #if os.path.exists(os.path.join(file_path , "labels" )):
    #    os.mkdir(os.path.join(file_path , "labels1"  ))
    #else:
    #     os.mkdir(os.path.join(file_path , "labels"  ))
         

    def label_files_to_dict(labels_path):
        label_file_list = glob(os.path.join(labels_path ,'*.txt'))
        label_dict = {}
        for label_file in label_file_list:
            label_file_name = label_file.split('/')[-1].split('.')[0]   #todo : make a function to sep
            with open(label_file_name , 'r') as file:
                lables = file.readlines()
                label_dict["label_file_name"] = lables
        return label_dict


    model = YOLO(model_path)



    count = 0 
    file_tif  = glob(os.path.join(dataset_path ,'*.tif'))
    file_jpg = glob(os.path.join(dataset_path ,'*.jpg'))
    file = file_tif + file_jpg

    start = time()

    for image in file:
        image_name , extension = str(os.path.split(image)[-1]).split('.')
        label_file = os.path.join(result_path , "labels" , f"{image_name}.txt")
        count = count + 1
        cv_image = cv2.imread(image)
        h , w , _= cv_image.shape
        pred_start_t = time()
        results = model.predict(cv_image, save=False, conf=0.4 , device=1 )
        pred_duration = time() - pred_start_t
        print(pred_duration)
        boxes = results[0].boxes
        res_file = os.path.join(result_path , "res_sgd" , f"{image_name}.{extension}")

        if(len(boxes.cls)==0):
            res_file = os.path.join(result_path , "res_test" , f"back_{image_name}.{extension}")
            cv2.imwrite(res_file, cv_image)
        for index in range(len(boxes.cls)):
            x1 = boxes.xyxyn[index][0].item() * w
            y1 =  boxes.xyxyn[index][1].item()* h
            x2 = boxes.xyxyn[index][2].item()* w
            y2 = boxes.xyxyn[index][3].item()* h
            start_point = (int(x1), int(y1))
            end_point = (int(x2), int(y2))

            annotated_image = cv2.rectangle(cv_image , start_point , end_point , (0,0,255) , 10)
            cv2.imwrite(res_file, annotated_image)

    time_taken = time() - start
    fps = count / time_taken
    print(f"time taken to predict {count} number images is {time_taken}")
    print(f"fps: {fps}")
            

class Annotate:
    pass
        
