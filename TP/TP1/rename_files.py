import os
import shutil

path = "./dct_db/"
new_path = "./new/"


for i, filename in enumerate(os.listdir(path)):
    shutil.copyfile(
    	path + filename,
     	new_path + str(i) + '.tif'
    )
    print("wrote file:", new_path + str(i) + '.tif')
