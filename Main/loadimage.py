import pandas as pd
from tqdm import tqdm
import os
import cv2
import pickle
import random
import numpy as np
from glob import glob

#training data
train = pd.read_csv('D:/user/Documents/Skripsi/Dataset/fix/train_newest8.csv')

#path
image_path =  'C:/train_tests_crop/' 
X_n_y_path = 'C:/train/'

# creating empty list
train_image_data = []
temp_image = []
image_count = 0

#load image
print("[INFO] load image ...")
for i in tqdm(range(train.shape[0])):
    if not train['class'][i]:
        continue
    if not os.path.exists(os.path.join(image_path, train['image'][i])):
        continue
    # loading the image rgb
    img = cv2.imread(os.path.join(image_path, train['image'][i]), cv2.IMREAD_GRAYSCALE)
    
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
    # img2 = np.zeros(shape=[224, 224, 3], dtype=np.uint8)
    # img2[:,:,0] = frame
    # img2[:,:,1] = frame
    # img2[:,:,2] = frame 
    # appending the image and the label into the list
    temp_image.append(frame)
    image_count = image_count + 1
    if image_count > 9:
        train_image_data.append([np.concatenate(temp_image, axis=2), train['class'][i]])
        temp_image.clear()
        image_count = 0
    # train_image_data.append([img2, train['class'][i]])
    del img 
del train

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
f = open(os.path.join(X_n_y_path, 'x6.pickle'), "wb")
f.write(pickle.dumps(X))
f.close()

f = open(os.path.join(X_n_y_path, 'y6.pickle'), "wb")
f.write(pickle.dumps(y))
f.close()