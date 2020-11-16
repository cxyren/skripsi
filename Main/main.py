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

#training
train=pd.read_csv('D:/user/Documents/Skripsi/Dataset/train_new.csv')

