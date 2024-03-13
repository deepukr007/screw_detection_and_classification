import os 

def rename_jpg_files(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(".jpg"):
            old_path = os.path.join(directory, filename)
            new_filename = filename.lower()
            new_path = os.path.join(directory, new_filename)
            
            os.rename(old_path, new_path)
            print(f'Renamed: {old_path} to {new_path}')

target_directory = "/home/krishnar@iff.intern/thesis/datasets/crs/validation/images"

rename_jpg_files(target_directory)

