import cv2
import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import ffmpy

#function
def convert_avi_to_mp4(avi_file_path, output_name):
    ff = ffmpy.FFmpeg(
        inputs={avi_file_path: None},
        outputs={output_name: None}
        )
    ff.run()

train = pd.read_csv('D:/user/Documents/Skripsi/Dataset/fix/RGB_newest3.csv')

raw_path = 'D:/user/Documents/Skripsi/Dataset/RGB-raw/nturgb+d_rgb/'
mp4_path = 'D:/user/Documents/Skripsi/Dataset/mp4/'

for i in tqdm(range(train.shape[0])):
    videoFile = train['video'][i].split('.')[0]
    print(videoFile)
    if not os.path.isfile(os.path.join(mp4_path, videoFile + '.mp4')):
        convert_avi_to_mp4(os.path.join(raw_path, videoFile + ".avi"),os.path.join(mp4_path, videoFile + ".mp4"))