
def get_datset(name):
    data_conf = {
            "crs" : '/home/krishnar@iff.intern/thesis/screw_detection/yolov8/dataset_conf/screw_data.yaml', 
            "weee" : '/home/krishnar@iff.intern/thesis/screw_detection/yolov8/dataset_conf/WEEE_data.yaml',
            "iff": '/home/krishnar@iff.intern/thesis/screw_detection/yolov8/dataset_conf/iff_screw_data.yaml'
        }
    return data_conf[name]