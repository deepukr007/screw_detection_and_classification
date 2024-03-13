from ultralytics import YOLO
from utils import get_datset , get_model

if __name__ == "__main__":

    epochs = 50
    patience = epochs
    model_name = "yolov8n"
    model_path = "/home/krishnar@iff.intern/thesis/screw_detection/src/ultralytics/yolov8/crs_yolo/n_epochs_100_batch_16_imgsz16643/weights/best.pt"

    model = get_model(model_name)
    dataset = get_datset("crs")

    #config
    imgsz=1664
    batch=16

    model = YOLO(model_path)
    results = model.train(data=dataset , epochs=epochs,  patience=patience ,workers=8,cos_lr=True, optimizer='SGD' ,lr0=0.01 , lrf=0.01, 
                            device='1' , batch=batch, imgsz=imgsz  , project="crs_yolo_sgd_resume", name=f"{model_name[-1]}_epochs_{epochs}_batch_{batch}_imgsz{imgsz}" )
  
    print(results)
    print(results.results_dict['metrics/mAP50(B)'])