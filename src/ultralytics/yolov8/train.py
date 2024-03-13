from ultralytics import YOLO
from utils import get_datset , get_model

if __name__ == "__main__":

    epochs = 250
    patience = epochs
    model_name = "yolov8n-seg"
    
    #model = get_model(model_name)
    dataset = get_datset("weee")

    #config
    imgsz=1280
    batch=30
    optimizer='AdamW' 
    lr0=0.03 
    lrf=0.01

    model = YOLO(model_name)
    results = model.train(data=dataset , epochs=epochs,  patience=patience ,workers=8, 
                            device='1' , batch=batch, imgsz=imgsz  , project="crs_yolo_adam", name=f"{model_name[-1]}_epochs_{epochs}_batch_{batch}_imgsz{imgsz}" )
  
    print(results)
    print(results.results_dict['metrics/mAP50(B)'])