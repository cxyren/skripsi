import cv2
import os

path = 'D:/user/Documents/Skripsi/Github-Program/Etc/'
vidcap = cv2.VideoCapture(os.path.join(path,"SampleVideo_1280x720_1mb.mp4"))
success,image = vidcap.read()
count = 0

while success:
  path = 'D:/user/Documents/Skripsi/Github-Program/Etc/Tes/'
  cv2.imwrite(os.path.join(path, "frame%d.jpg" % count), image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1