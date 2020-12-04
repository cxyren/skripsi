import cv2
import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import math

#function
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

#path
skeleton_path = 'D:/user/Documents/Skripsi/Dataset/ntu-skeleton/skeletons/'
missing_skeleton_path = 'D:/user/Documents/Skripsi/Dataset/ntu_rgbd_missings.txt'
dest_path = 'C:/train_test2/'

#class that been used
name_class = pd.read_csv('D:/user/Documents/Skripsi/Dataset/class_name_new.csv')

#used setup
setup_num = dict()
setup_num['S003'] = True
setup_num['S004'] = True
setup_num['S005'] = True
setup_num['S006'] = True
setup_num['S008'] = True
setup_num['S009'] = True

#used class
class_code = dict()
for i in range(name_class.shape[0]):
    class_code[name_class['code'][i]] = name_class['name'][i]

#array of joint that connected
connecting_joint = [2, 1, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 8, 8, 12, 12]

#skeleton missing files
missing_files = load_missing_file(missing_skeleton_path)

#create empty list
skeleton_image = []
skeleton_name = []
skeleton_class = []
spine_mid_x = []
spine_mid_y = []

#get all skeleton
skeleton = glob(os.path.join(skeleton_path, '*'))

#storing the frames
for i in tqdm(range(len(skeleton))):
    #check if skeleton not valid
    check = False   

    #get skeleton name 
    skeleton_file_name = skeleton[i].split('/')[6][10:]

    #check if skeleton file is missing or not
    if skeleton_file_name in missing_files:
        continue
    
    #check if skeleton in the setup dict
    if skeleton_file_name[:4] not in setup_num:
        continue

    #check if skeleton not in class dict
    if skeleton_file_name.split('.')[0][-4:] not in class_code:
        continue

    #read skeleton file
    bodyinfo = read_skeleton_file(os.path.join(skeleton_path, skeleton_file_name))

    count = math.floor((len(bodyinfo) - 5) *1.0 / 10)

    framecount = 0
    
    #loop for in frame
    for j in range(len(bodyinfo)): 
        #get 10 frame 
        
        if (j - 4) % count == 0 :            
            framecount = framecount + 1
            if framecount > 10:
                break
            #make blank images
            frame = np.zeros(shape=[1080, 1920, 3], dtype=np.uint8)
            color = tuple(reversed([0,0,0]))
            frame[:] = color
            #loop in skeleton joint
            for l in range(25):
                try:
                    # red for line
                    rv = 255
                    gv = 0
                    bv = 0
                    #search for joint that connect
                    m = connecting_joint[l] - 1
                    #get joint x and y
                    joint = bodyinfo[j][0]['joints'][l]
                    dx = np.int32(round(float(joint['colorX'])))
                    dy = np.int32(round(float(joint['colorY'])))
                    joint2 = bodyinfo[j][0]['joints'][m]
                    dx2 = np.int32(round(float(joint2['colorX'])))
                    dy2 = np.int32(round(float(joint2['colorY'])))
                    #get pixel for the line 
                    line = bresenham_line(dx, dy, dx2, dy2)
                    #write line per pixel
                    for n in range(len(line)):
                        dx = line[n][0]
                        dy = line[n][1]
                        frame = cv2.circle(frame, (dx, dy), radius=2, color=(bv, gv, rv), thickness=-1)

                    #green color for points/ joints
                    rv = 0
                    gv = 255
                    bv = 0
                    #get x and y
                    joint = bodyinfo[j][0]['joints'][l]
                    dx = np.int32(round(float(joint['colorX'])))
                    dy = np.int32(round(float(joint['colorY'])))
                    #write joint
                    frame = cv2.circle(frame, (dx, dy), radius=5, color=(bv, gv, rv), thickness=-1)
                except:
                    #if theres error then break
                    check = True
                    break
            #if theres error then break
            if check:
                break
            joint = bodyinfo[j][0]['joints'][1]
            spine_mid_x.append(joint['colorX'])
            spine_mid_y.append(joint['colorY'])
            #save label and name
            skeleton_name.append(skeleton_file_name.split('.')[0].split('_')[0])
            skeleton_image.append(skeleton_file_name.split('.')[0].split('_')[0] +"_frame%d.jpg" % j)
            skeleton_class.append(class_code.get(skeleton_file_name.split('.')[0][-4:]))
            #save file
            filename = os.path.join(dest_path,  skeleton_file_name.split('.')[0].split('_')[0] +"_frame%d.jpg" % j)
            frame = cv2.resize(frame, (int(frame.shape[1] * 0.3), int(frame.shape[0] * 0.3)))
            cv2.imwrite(filename, frame)


# storing the images and their class in a dataframe
df = pd.DataFrame()
df['skeleton'] =skeleton_name
df['image'] = skeleton_image
df['class'] = skeleton_class
df['spine_mid_X'] = spine_mid_x
df['spine_mid_Y'] = spine_mid_y

print('[INFO]SAVING INTO CSV...')
# converting the dataframe into csv file 
df.to_csv('D:/user/Documents/Skripsi/Dataset/fix/train_newest9.csv', header=True, index=False)