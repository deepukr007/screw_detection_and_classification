from glob import glob
import os
import cv2


label_path = "/home/krishnar@iff.intern/thesis/screw_detection/yolov8/labels"

count_images_with_objects =0 
count_total_objects = 0 

conf_threshold = 0 


for label_file in glob(os.path.join(label_path , '*.txt')):
    count_images_with_objects = count_images_with_objects +1 
    with open(label_file , 'r') as file:
        objects = file.readlines() 
        count_total_objects = count_total_objects + len(objects)
        for box in objects:
            start_point = (int(box[1]), int(box[2]))
            end_point = (int(box[3]), int(box[4]))
            np_image = cv2.rectangle(result_item.image, start_point, end_point, (0,0,255), 10) 
cv2.imwrite(os.path.join(output , file_name) , np_image)

print("Number of labelled images : " , count_images_with_objects)
print("Number of total instances: "  , count_total_objects)


