import pandas as pd
from tqdm import tqdm
import os
import cv2
import pickle
import random
import numpy as np
from glob import glob

#training data
train = pd.read_csv('D:/user/Documents/Skripsi/Dataset/fix/train_newest15.csv')

#path
# image_path =  'C:/new_train_crop/' 
X_n_y_path = 'C:/train/'

# creating empty list
train_image_data = []
temp_image = []
image_count = 0
temp_name = dict()

#load image
print("[INFO] load image ...")
for i in tqdm(range(train.shape[0])):
    if not train['class'][i]:
        continue
    if not os.path.exists(os.path.join(train['dir'][i], train['image'][i])):
        continue

    # print(image_count)
    # print(train['skeleton'][i])
    # print(temp_name)
    # if image_count == 0:
    #     temp_name[train['skeleton'][i]] = True

    # if train['skeleton'][i] not in temp_name:
    #     # print(image_count)
    #     # print(train['skeleton'][i])
    #     # print(temp_name)
    #     temp_image.clear()
    #     image_count = 0
    #     temp_name.clear()
    #     continue
    
    image_count = image_count + 1

    # loading the image
    img = cv2.imread(os.path.join(train['dir'][i], train['image'][i]), cv2.IMREAD_GRAYSCALE)
    
    #making empty frame    
    frame = np.zeros(shape=[224, 224], dtype=np.uint8)
    if img.shape[1] < img.shape[0]:
        img = cv2.resize(img, (int(img.shape[1]*(224/img.shape[0])), 224))
        

        #relocating image
        center_x = frame.shape[1] / 2
        center_x2 = img.shape[1] / 2
        
        frame[int(0):int(224), int(center_x - center_x2):int(center_x + center_x2)] = img
    else:
        img = cv2.resize(img, (224, int(img.shape[0]*(224/img.shape[1]))))

        #relocating image
        center_y = frame.shape[0] / 2
        center_y2 = img.shape[0] / 2
        
        frame[int(center_y - center_y2):int(center_y + center_y2), int(0):int(224)] = img
        
    #resize image
    frame = cv2.resize(frame, (224, 224))
    frame = np.expand_dims(frame, axis=2)  
    temp_image.append(frame)
    if image_count >= 10:
        # print(train['skeleton'][i])
        # print(temp_name)
        train_image_data.append([np.concatenate(temp_image, axis=2), train['class'][i]])
        temp_image.clear()
        image_count = 0
        temp_name.clear()
    del img 
del train


print(len(train_image_data))
print(train_image_data[0][0].shape)
#shuffle the data
print("[INFO] shuffle data ...")    
random.shuffle(train_image_data)

#save to x and y
X = []
y = []
for image, label in train_image_data:
    X.append(image)
    y.append(label)

#saving data
print("[INFO] saving image data ...")
f = open(os.path.join(X_n_y_path, 'new_trainx4.pickle'), "wb")
f.write(pickle.dumps(X))
f.close()

f = open(os.path.join(X_n_y_path, 'new_trainy4.pickle'), "wb")
f.write(pickle.dumps(y))
f.close()