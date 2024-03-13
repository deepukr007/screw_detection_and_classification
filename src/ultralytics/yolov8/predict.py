from ultralytics import YOLO
import cv2 
from glob import glob
import os




if __name__ == "__main__":
    model_path = "/home/krishnar@iff.intern/thesis/screw_detection/src/ultralytics/yolov8/crs_yolo_sgd/n_epochs_100_batch_12_imgsz19202/weights/best.pt"
    dataset_path = "datasets/crs/validation/images"
    file_path = os.path.dirname(os.path.abspath(__file__))

    if os.path.exists(os.path.join(file_path , "labels" )):
        os.mkdir(os.path.join(file_path , "labels1"  ))
    else:
         os.mkdir(os.path.join(file_path , "labels"  ))
         


    model = YOLO(model_path)


    count = 0 
    file_tif  = glob(os.path.join(dataset_path ,'*.tif'))
    file_jpg = glob(os.path.join(dataset_path ,'*.jpg'))
    file = file_tif + file_jpg

    for image in file:
        image_name = str(os.path.split(image)[-1]).split('.')[0] 
        label_file = os.path.join(file_path , "labels" , f"{image_name}.txt")
        count = count + 1
        cv_image = cv2.imread(image)
        results = model.predict(cv_image, save=False, conf=0.1 , device = 1)
        boxes = results[0].boxes
        for index in range(len(boxes.cls)):
            x = boxes.xywhn[index][0].item()
            y = boxes.xywhn[index][1].item()
            w = boxes.xywhn[index][2].item()
            h = boxes.xywhn[index][3].item()
            string = f"{int(boxes.cls[index])} {x} {y} {w} {h} {boxes.conf[index]} \n"
            file = open(f"{label_file}" , 'a+') 
            file.write(string)
            file.close()
        
