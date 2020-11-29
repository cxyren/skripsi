import pandas as pd
from tqdm import tqdm
import os
import cv2
import pickle
import random

#training data
train = pd.read_csv('D:/user/Documents/Skripsi/Dataset/fix/train_newest4.csv')

#path
image_path =  'C:/train_image/' 
X_n_y_path = 'C:/train/'

# creating empty list
train_image_data = []

#load image
print("[INFO] load image ...")
for i in tqdm(range(train.shape[0])):
    top = 9999
    bot = -1
    left = 9999
    right = -1
    if not train['class'][i]:
        continue
    # loading the image rgb
    img = cv2.imread(os.path.join(image_path, train['image'][i]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # get skeleton area
    for j in range(img.shape[1]):
        for k in range(img.shape[0]):
            if all(k > 0 for k in img[k,j]):
                if j < top:
                    top = j
                if j > bot:
                    bot = j
                if i < left:
                    left = i
                if i > right:
                    right = i
    #crop image
    crop_img = img[int(top):int(bot), int(left):int(right)]
    #resize image
    img = cv2.resize(img, (224, 224))
    # appending the image and the label into the list
    train_image_data.append([img, train['class'][i]])
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
f = open(os.path.join(X_n_y_path, 'x2.pickle'), "wb")
f.write(pickle.dumps(X))
f.close()

f = open(os.path.join(X_n_y_path, 'y2.pickle'), "wb")
f.write(pickle.dumps(y))
f.close()