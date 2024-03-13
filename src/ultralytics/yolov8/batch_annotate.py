from utils import *
import os 



image_in_path = "/home/krishnar@iff.intern/thesis/datasets/crs/validation/images"
label_in_path = "/home/krishnar@iff.intern/thesis/datasets/crs/validation/labels"


image_out_path = "/home/krishnar@iff.intern/thesis/pred_res/images_labelled"

if  os.path.exists(image_out_path) == False:
    os.mkdir(image_out_path )
labels_dict  = label_files_to_dict(label_in_path)
batch_annotate_save(labels_dict , image_in_path , image_out_path)
print(len(labels_dict))

images = background_images(image_in_path , label_in_path)
print(images)