import cv2
import os
import math
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing import image
import numpy as np
from keras.utils import np_utils
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer,Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
import ffmpy

#function
def convert_avi_to_mp4(avi_file_path, output_name):
    ff = ffmpy.FFmpeg(
        inputs={avi_file_path: None},
        outputs={output_name: None}
        )
    ff.run()

#preprocessing video from csv and convert into frame
train=pd.read_csv('D:/user/Documents/Skripsi/Dataset/RGB.csv')
train.head()

train_image = []

raw_path = 'D:/user/Documents/Skripsi/Dataset/RGB-raw/'
mp4_path = 'D:/user/Documents/Skripsi/Dataset/RGB-mp4/'

# storing the frames from training videos
for i in tqdm(range(1)):
    count = 0
    videoFile = train['video'][i].split('.')[0]
    convert_avi_to_mp4(os.path.join(raw_path, videoFile + ".avi"),os.path.join(mp4_path, videoFile + ".mp4"))
    cap = cv2.VideoCapture(os.path.join(mp4_path, videoFile + ".mp4"))   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    x=1
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            dest_path = 'D:/user/Documents/Skripsi/Dataset/train/'
            filename = os.path.join(dest_path,  videoFile.split('_')[0] +"_frame%d.jpg" % count);count+=1
            cv2.imwrite(filename, frame)
            print("success make frame %d" %count)
    cap.release()

# getting the names of all the images
images = glob("D:/user/Documents/Skripsi/Dataset/train/*")
train_image = []
train_class = []
for i in tqdm(range(len(images))):
    # creating the image name
    _nameimage = images[i].split('/')[6]
    train_image.append(_nameimage[4:])
    # creating the class of image 
    _class = _nameimage.split('_')[0][-4:]
    print(_class)
    if _class == "A001" :
      _nameclass = "Drink"
    elif _class == "A008":
      _nameclass = "Sit"
    elif _class == "A009":
      _nameclass = "Stand Up"
    elif _class == "A011":
      _nameclass = "Reading"
    else:
      _nameclass = "Playing with phone"
    print(_nameclass)
    train_class.append(_nameclass)
    
# storing the images and their class in a dataframe
train_data = pd.DataFrame()
train_data['image'] = train_image
train_data['class'] = train_class

# converting the dataframe into csv file 
train_data.to_csv('D:/user/Documents/Skripsi/Dataset/train_new.csv',header=True, index=False)

#training
