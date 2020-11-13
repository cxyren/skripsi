import cv2
import os
import ffmpy
from glob import glob
import pandas as pd
from tqdm import tqdm

videoFile = "sampleA001_rgb"
path = 'D:/user/Documents/Skripsi/Github-Program/Etc/'

ff = ffmpy.FFmpeg(  
    inputs={videoFile + '.avi': None},
    outputs={'video/' + videoFile + '.mp4': None}
    )
ff.run()

vidcap = cv2.VideoCapture(os.path.join(path + 'video/',videoFile + ".mp4"))
success,image = vidcap.read()
count = 0

while success:
  path = 'D:/user/Documents/Skripsi/Github-Program/Etc/Tes/'
  cv2.imwrite(os.path.join(path, videoFile.split('_')[0] + "_frame%d.jpg" % count), image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1

# getting the names of all the images
images = glob("D:/user/Documents/Skripsi/Github-Program/Etc/Tes/*", )
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
train_data.to_csv('D:/user/Documents/Skripsi/Github-Program/Etc/train_new.csv',header=True, index=False)