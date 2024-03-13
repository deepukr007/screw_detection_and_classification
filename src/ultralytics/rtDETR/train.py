from ultralytics import RTDETR
import torch
from utils import get_datset


model = RTDETR('rtdetr-l.pt')

epochs = 500
patience = epochs
        
dataset = get_datset("crs")


# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data=dataset, epochs=100, imgsz=640 , project="train_screw_detr" , name="epoch_100_batch_20_imgz_640" , batch=20 , device=1 )  