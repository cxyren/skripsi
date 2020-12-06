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
train_path = 'C:/new_train/'
test_path = 'C:/new_test'

#class that been used
name_class = pd.read_csv('D:/user/Documents/Skripsi/Dataset/class_name_new_new.csv')

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

for i in range(2):
    skeleton_image.append([])
    skeleton_name.append([])
    skeleton_class.append([])
    spine_mid_x.append([])
    spine_mid_y.append([])

#get all skeleton
skeleton = glob(os.path.join(skeleton_path, '*'))

count = 0
print('[INFO] GET COUNT')
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

    count = count + 1

print('TOTAL: %i'%count) #4368
print('TRAIN : %i'%round(count*0.8)) #3494
print('TEST : %i'%round(count*0.2)) #874

count_train = 0
count_test = 0

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

    count = round((len(bodyinfo) - 5) *1.0 / 10)

    framecount = 0
    
    #loop for in frame
    for j in range(len(bodyinfo)): 
        #get 10 frame 
        
        if (j - 4) % count == 0 :            
            framecount = framecount + 1
            if framecount > 10:
                print(framecount)
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
            
            frame = cv2.resize(frame, (int(frame.shape[1] * 0.3), int(frame.shape[0] * 0.3)))
            # left = 9999
            # right = -1
            # bot = 0

            # for m in range(frame.shape[1]):
            #     for n in range(frame.shape[0]):
            #         if np.any(frame[n,m]):
            #             if n > bot:
            #                 bot = n
            #             if m < left:
            #                 left = m
            #             if m > right:
            #                 right = m

            # right = right + 2
            # left = left - 2
            # bot = bot + 2
            # top = bot - 175
            # if top < 0 :
            #     top = 0

            # # print(frame.shape)
            # # print('top:%i,bot:%i,left:%i,right:%i'%(top,bot,left,right))

            # frame = frame[int(top):int(bot), int(left):int(right)]

            joint = bodyinfo[j][0]['joints'][1]
            if count_train < 3494:
                spine_mid_x[0].append(joint['colorX'])
                spine_mid_y[0].append(joint['colorY'])
                #save label and name
                skeleton_name[0].append(skeleton_file_name.split('.')[0].split('_')[0])
                skeleton_image[0].append(skeleton_file_name.split('.')[0].split('_')[0] +"_frame%d.jpg" % j)
                skeleton_class[0].append(class_code.get(skeleton_file_name.split('.')[0][-4:]))
                #save file
                filename = os.path.join(train_path,  skeleton_file_name.split('.')[0].split('_')[0] +"_frame%d.jpg" % j)
                
                cv2.imwrite(filename, frame)
            else:
                spine_mid_x[1].append(joint['colorX'])
                spine_mid_y[1].append(joint['colorY'])
                #save label and name
                skeleton_name[1].append(skeleton_file_name.split('.')[0].split('_')[0])
                skeleton_image[1].append(skeleton_file_name.split('.')[0].split('_')[0] +"_frame%d.jpg" % j)
                skeleton_class[1].append(class_code.get(skeleton_file_name.split('.')[0][-4:]))
                #save file
                filename = os.path.join(test_path,  skeleton_file_name.split('.')[0].split('_')[0] +"_frame%d.jpg" % j)
                cv2.imwrite(filename, frame)
    if count_train < 3494:
        count_train = count_train + 1
    else:        
        count_test = count_test + 1

print('TRAIN : %i'%count_train) #3494
print('TEST : %i'%count_test) #874

print('[INFO]SAVING INTO CSV...')
# storing the images and their class in a dataframe
df = pd.DataFrame()
df['skeleton'] =skeleton_name[0]
df['image'] = skeleton_image[0]
df['class'] = skeleton_class[0]
df['spine_mid_X'] = spine_mid_x[0]
df['spine_mid_Y'] = spine_mid_y[0]

# converting the dataframe into csv file 
df.to_csv('D:/user/Documents/Skripsi/Dataset/fix/train_newest10.csv', header=True, index=False)

# storing the images and their class in a dataframe
df = pd.DataFrame()
df['skeleton'] =skeleton_name[1]
df['image'] = skeleton_image[1]
df['class'] = skeleton_class[1]
df['spine_mid_X'] = spine_mid_x[1]
df['spine_mid_Y'] = spine_mid_y[1]

# converting the dataframe into csv file 
df.to_csv('D:/user/Documents/Skripsi/Dataset/fix/test_newest10.csv', header=True, index=False)