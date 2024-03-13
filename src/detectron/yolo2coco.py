from pathlib import Path
from detectron2.structures import BoxMode
import cv2
from detectron2.data.catalog import DatasetCatalog
from detectron2.data import MetadataCatalog
from glob import glob
import os 

def create_data_pairs(input_path, detectron_img_path, detectron_annot_path, dir_type = 'train'):

    #img_paths = Path(input_path + dir_type + '/images/').glob('*.jpg')
    img_paths = glob(os.path.join(input_path + dir_type + '/images/','*.jpg'))

    pairs = []
    for img_path in img_paths:

        file_name_tmp = str(img_path).split('/')[-1].split('.')
        file_name_tmp.pop(-1)
        file_name = '.'.join((file_name_tmp))

        label_path = Path(input_path + dir_type + '/labels/' + file_name + '.txt')

        if label_path.is_file():

            line_img = detectron_img_path + dir_type + '/images/'+ file_name + '.jpg'
            line_annot = detectron_annot_path + dir_type + '/labels/' + file_name + '.txt'
            pairs.append([line_img, line_annot])

    return pairs


def create_coco_format(data_pairs):
    
    data_list = []

    for i, path in enumerate(data_pairs):
        
        filename = path[0]

        img_h, img_w = cv2.imread(filename).shape[:2]

        img_item = {}
        img_item['file_name'] = filename
        img_item['image_id'] = i
        img_item['height']= img_h
        img_item['width']= img_w

        print(str(i), filename)


        annotations = []
        with open(path[1]) as annot_file:
            lines = annot_file.readlines()
            for line in lines:
                if line[-1]=="\n":
                  box = line[:-1].split(' ')
                else:
                  box = line.split(' ')

                class_id = box[0]
                x_c = float(box[1])
                y_c = float(box[2])
                width = float(box[3])
                height = float(box[4])

                x1 = (x_c - (width/2)) * img_w
                y1 = (y_c - (height/2)) * img_h
                x2 = (x_c + (width/2)) * img_w
                y2 = (y_c + (height/2)) * img_h

                annotation = {
                    "bbox": list(map(float,[x1, y1, x2, y2])),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": int(class_id),
                    "iscrowd": 0
                }
                annotations.append(annotation)
            img_item["annotations"] = annotations
        data_list.append(img_item)
    return data_list 




def register_dataset():
    input_path = '/home/krishnar@iff.intern/thesis/datasets/crs/'
    detectron_img_path = '/home/krishnar@iff.intern/thesis/datasets/crs/'
    detectron_annot_path = '/home/krishnar@iff.intern/thesis/datasets/crs/'

    if 
    train = create_data_pairs(input_path, detectron_img_path,detectron_annot_path,  'train')
    val = create_data_pairs(input_path, detectron_img_path, detectron_annot_path,'validation')

    train_list = create_coco_format(train)
    val_list = create_coco_format(val)

    for catalog_name, file_annots in [("train", train_list), ("val", val_list)]:
        DatasetCatalog.register(catalog_name, lambda file_annots = file_annots: file_annots)
        MetadataCatalog.get(catalog_name).set(thing_classes=['crs'])

    metadata = MetadataCatalog.get("train") 


