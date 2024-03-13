import os
from glob import glob

path = "/home/data/iDear/Own2_processsed"

for file in glob(os.path.join(path , "*")):
    extension = file.split("/")[-1].split(".")[1]
    if extension  not in ["tif" , "jpg" , "jpeg" ,"png"]:
        print(f"removing {file}")
        os.remove(file)