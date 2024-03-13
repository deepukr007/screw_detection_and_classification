import os 
from glob import glob

import cv2 
from ultralytics.utils.plotting import Annotator


models_path = "/home/krishnar@iff.intern/thesis/screw_detection/src/ultralytics/models/"
yolo_v8_nano = os.path.join(models_path, "base/yolov8n.pt")
yolo_v8_small = os.path.join(models_path ,"base/yolov8s.pt")
yolo_v8_medium = os.path.join(models_path , "base/yolov8m.pt")

dataset_conf_path = "/home/krishnar@iff.intern/thesis/screw_detection/src/ultralytics/dataset_conf"
    
crs_conf = os.path.join(dataset_conf_path , "screw_data.yaml" )
weee_conf = os.path.join(dataset_conf_path , "WEEE_data.yaml")
iff_conf = os.path.join(dataset_conf_path , "iff_screw_data.yaml" )

data_conf = {
        "crs":crs_conf,
        "weee":weee_conf,
        "iff" :iff_conf
    }

models = {
        "yolov8n": yolo_v8_nano,
        "yolov8m":yolo_v8_medium ,
        "yolov8s": yolo_v8_small
    }



def get_datset(name):  
    return data_conf[name]


def get_model(name):
    return models[name]
  
def xywh2xyxy(x , y , w , h):
    x1n = x - (w / 2)
    y1n = y + (h / 2)
    x2n = x + (w / 2)
    y2n = y - (h / 2)

    return x1n,x2n,y1n ,y2n

def get_filename_and_extension(path):
    filename = path.split('/')[-1]
    file_id , extension = filename.split('.')[:2]
    return file_id , extension


def label_files_to_dict(labels_path):
    label_file_list = glob(os.path.join(labels_path ,'*.txt'))
    label_dict = {}
    for label_file in label_file_list:
         label_file_name , _ = get_filename_and_extension(label_file)  #todo : make a function to sep
         with open(label_file , 'r') as file:
            lables = file.readlines()
            label_dict[label_file_name] = lables
    return label_dict


def batch_annotate_save(labels_dict , images_in_path , images_out_path):
    for image in glob(os.path.join(images_in_path , '*.jpg')):
        file_id , extension = get_filename_and_extension(image)
        output_image = os.path.join(images_out_path , file_id+'.jpg')
        objects = labels_dict.get(file_id ,None)
        if objects is not None:
                cv_image = cv2.imread(image)
                i_h , i_w , _= cv_image.shape
                for instance in objects:
                    values = instance.strip('\n').split(" ")
                    x1n ,x2n , y1n, y2n  = xywh2xyxy(float(values[1]) , float(values[2]) , float(values[3]) , float(values[4]))
                    start_point = (int(x1n*i_w ), int(y1n*i_h))
                    end_point = (int(x2n*i_w) , int(y2n*i_h))
                    cv_image_annotated = cv2.rectangle(cv_image , start_point , end_point , (0,0,255) , 10)
                cv2.imwrite(output_image, cv_image_annotated)



def background_images(img_path , lbl_path):
    images = [get_filename_and_extension(x)[0] for x in glob(os.path.join(img_path ,"*.jpg"))]
    labels = [get_filename_and_extension(x)[0] for x in glob(os.path.join(lbl_path ,"*.txt"))]
    background_images = [x for x in images if x not in labels]
    return background_images

    

def draw_box(img , box , label):
    annotator = Annotator(img)
    annotator.box_label(box , label)
    return img



