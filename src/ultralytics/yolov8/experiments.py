from ultralytics import YOLO
from time import time
from utils import get_datset


if __name__ == "__main__":

    start = time()

    dataset = get_datset("crs")
    cont = True
    batch_min=4
    imgz_min=640

    batch_list = [1 ,2 , 4,  8, 16, 32, 64, 128 ]
    imgsz_list = [4480 , 3840 ,2560 , 1920 , 1280 , 640 ]
    batch= batch_min
    imgsz = imgz_min
    for imgsz in imgsz_list[3:]:
        for batch in batch_list[2:]:
            try:
                model = YOLO("yolov8m.pt")
                print(f"batch_{batch}_imgz_{imgsz}")
                results = model.train(data=dataset, device=1,
                                    epochs=1, batch=batch, imgsz=imgsz , workers=8, project="Yolo_exp_m" , name = f"batch_{batch}_imgz_{imgsz}")
            except Exception as e:
                print(e)
                break    

    end = time() - start
    print(end)
    