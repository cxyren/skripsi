import cv2
import os
import math
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

def read_skeleton_file(filename):
    file = open(filename)
    framecount = file.readline()

    bodyinfo = []
    for i in range(int(framecount)):
        bodycount = file.readline()
        bodies = []
        for j in range(int(bodycount)):
            arraynum = file.readline().split()
            body = {
                "bodyID": arraynum[0],
                "clipedEdges": arraynum[1],
                "handLeftConfidence": arraynum[2],
                "handLeftState": arraynum[3],
                "handRightConfidence": arraynum[4],
                "handRightState": arraynum[5],
                "isResticted": arraynum[6],
                "leanX": arraynum[7],
                "leanY": arraynum[8],
                "trackingState": arraynum[9],
                "jointCount": file.readline(),
                "joints": []
            }
            for k in range(int(body["jointCount"])):
                jointinfo = file.readline().split()
                joint={
                    "x": jointinfo[0],
                    "y": jointinfo[1],
                    "z": jointinfo[2],
                    "depthX": jointinfo[3],
                    "depthY": jointinfo[4],
                    "colorX": jointinfo[5],
                    "colorY": jointinfo[6],
                    "orientationW": jointinfo[7],
                    "orientationX": jointinfo[8],
                    "orientationY": jointinfo[9],
                    "orientationZ": jointinfo[10],
                    "trackingState": jointinfo[11]
                }
                body["joints"].append(joint)
            bodies.append(body)
        bodyinfo.append(bodies)
    file.close()
    return bodyinfo

def bresenham_line(x0, y0, x1, y1):
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0  
        x1, y1 = y1, x1

    switched = False
    if x0 > x1:
        switched = True
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    if y0 < y1: 
        ystep = 1
    else:
        ystep = -1

    deltax = x1 - x0
    deltay = abs(y1 - y0)
    error = -deltax / 2
    y = y0

    line = []    
    for x in range(x0, x1 + 1):
        if steep:
            line.append((y,x))
        else:
            line.append((x,y))

        error = error + deltay
        if error > 0:
            y = y + ystep
            error = error - deltax
    if switched:
        line.reverse()
    return line

def load_missing_file(path):
    missing_files = dict()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            if line not in missing_files:
                missing_files[line] = True 
    return missing_files 

#preprocessing video from csv and convert into frame
train=pd.read_csv('D:/user/Documents/Skripsi/Dataset/RGB.csv')
train.head()

train_image = []

raw_path = 'D:/user/Documents/Skripsi/Dataset/RGB-raw/nturgb+d_rgb/'
mp4_path = 'D:/user/Documents/Skripsi/Dataset/RGB-mp4/'
skeleton_path = 'D:/user/Documents/Skripsi/Dataset/ntu-skeleton/skeletons/'
missing_skeleton_path = 'D:/user/Documents/Skripsi/Dataset/ntu_rgbd_missings.txt'

connecting_joint = [2, 1, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 8, 8, 12, 12]

missing_files = load_missing_file(missing_skeleton_path)

# storing the frames from training videos
for i in tqdm(range(len(train))):
    #get name of files
    videoFile = train['video'][i].split('.')[0]
    #convert avi into mp4
    convert_avi_to_mp4(os.path.join(raw_path, videoFile + ".avi"),os.path.join(mp4_path, videoFile + ".mp4"))
    # capturing the video from the given path
    cap = cv2.VideoCapture(os.path.join(mp4_path, videoFile + ".mp4")) 
    #skeleton files
    skeletonFile = videoFile.split('_')[0]
    #chekc if skeleton file is missing or not
    if skeletonFile not in missing_files:
        #read skeleton file
        bodyinfo = read_skeleton_file(os.path.join(skeleton_path, skeletonFile + '.skeleton'))
        for j in range(len(bodyinfo)):
            if(cap.isOpened()):
                ret, frame = cap.read()
                if (ret != True):
                    break
                for k in range(len(bodyinfo[j])):
                    for l in range(int(bodyinfo[j][k]['jointCount'])):
                        # red for line
                        rv = 255
                        gv = 0
                        bv = 0
                        #search for joint that connect
                        m = connecting_joint[l] - 1
                        #get joint x and y
                        joint = bodyinfo[j][k]['joints'][l]
                        dx = np.int32(round(float(joint['colorX'])))
                        dy = np.int32(round(float(joint['colorY'])))
                        joint2 = bodyinfo[j][k]['joints'][m]
                        dx2 = np.int32(round(float(joint2['colorX'])))
                        dy2 = np.int32(round(float(joint2['colorY'])))
                        #get pixel for the line 
                        line = bresenham_line(dx, dy, dx2, dy2)
                        #write line per pixel
                        for n in range(len(line)):
                            dx = line[n][0]
                            dy = line[n][1]
                            frame = cv2.circle(frame, (dx, dy), radius=3, color=(bv, gv, rv), thickness=-1)

                        #green color for points/ joints
                        rv = 0
                        gv = 255
                        bv = 0
                        #get x and y
                        joint = bodyinfo[j][k]['joints'][l]
                        dx = np.int32(round(float(joint['colorX'])))
                        dy = np.int32(round(float(joint['colorY'])))
                        #write joint
                        frame = cv2.circle(frame, (dx, dy), radius=5, color=(bv, gv, rv), thickness=-1)

                dest_path = 'D:/user/Documents/Skripsi/Dataset/train/'
                filename = os.path.join(dest_path,  videoFile.split('_')[0] +"_frame%d.jpg" % j)
                cv2.imwrite(filename, frame)
    else:
        print("skeleton is missing")
        count = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if (ret != True):
                break
            dest_path = 'D:/user/Documents/Skripsi/Dataset/train/'
            filename = os.path.join(dest_path,  videoFile.split('_')[0] +"_frame%d.jpg" % count);count+=1
            cv2.imwrite(filename, frame)
    print("success processing video : %d" %i + 1)
    cap.release()

# getting the names of all the images
images = glob("D:/user/Documents/Skripsi/Dataset/train/*")
train_image = []
train_class = []
for i in tqdm(range(len(images))):
    # creating the image name
    _nameimage = images[i].split('/')[5]
    train_image.append(_nameimage[6:])
    # creating the class of image 
    _class = _nameimage.split('_')[0][-4:]
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
    train_class.append(_nameclass)
    
# storing the images and their class in a dataframe
train_data = pd.DataFrame()
train_data['image'] = train_image
train_data['class'] = train_class

# converting the dataframe into csv file 
train_data.to_csv('D:/user/Documents/Skripsi/Dataset/train_new.csv',header=True, index=False)