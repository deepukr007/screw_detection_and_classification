from ultralytics.data.converter import convert_coco

in_path = 
out_path = 

convert_coco(in_path, save_dir  = out_path ,  use_segments=False, use_keypoints=False, cls91to80=True)