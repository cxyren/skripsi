import pandas as pd
from glob import glob
import os
from tqdm import tqdm

dest_path = 'C:/train_image'
skiprow = 0

print('[INFO]PLACING LABEL INTO IMAGE...')
# getting the names of all the images
images = glob(os.path.join(dest_path, '*'))
name_class = pd.read_csv('D:/user/Documents/Skripsi/Dataset/class_name_new.csv', skiprows=skiprow)
train_image = []
train_class = []

class_count = [0]*name_class.shape[0] 
for i in tqdm(range(len(images))):
    # creating the image name
    _nameimage = images[i].split('/')[1]
    # creating the class of image 
    _class = _nameimage.split('_')[1][-4:]
    for j in range(name_class.shape[0]):
        if _class == name_class['code'][j]:
            if class_count[j] > 1471:
                break
            class_count[j] = class_count[j] + 1
            
            train_image.append(_nameimage[12:])
            train_class.append(name_class['name'][j])
            break
    
# storing the images and their class in a dataframe
train_data = pd.DataFrame()
train_data['image'] = train_image
train_data['class'] = train_class

print('[INFO]SAVING INTO CSV...')
# converting the dataframe into csv file 
train_data.to_csv('D:/user/Documents/Skripsi/Dataset/fix/train_newest4.csv', header=True, index=False)