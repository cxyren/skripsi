import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import os
import pickle

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

name_class = pd.read_csv('D:/user/Documents/Skripsi/Dataset/class_name_new.csv')
joint = []
class_code = dict()
for i in range(name_class.shape[0]):
    class_code[name_class['code'][i]] = name_class['name'][i]
    joint.append([])

list_skeleton = glob('D:/user/Documents/Skripsi/Dataset/ntu-skeleton/skeletons/*')

connecting_joint = [2, 1, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 8, 8, 12, 12]

class_count = 0
for i in tqdm(range(60)):
    if list_skeleton[i].split('.')[0][-4] not in class_code:
        continue
    
    bodyinfo = read_skeleton_file(list_skeleton[i])
    
    count = round((len(bodyinfo) - 5) *1.0 / 10)

    framecount = -1
    
    #loop for in frame
    for j in range(len(bodyinfo)): 
        #get 10 frame 
        if (j - 4) % count == 0 :            
            framecount += 1
            if framecount > 9:
                break
            joint[class_count].append([])
            for k in range(25):
                temp = bodyinfo[j][0]['joints'][k]
                joint[class_count][framecount].append([temp['colorX'], temp['colorY']])
            
    class_count += 1

f = open(os.path.join('C:/users/cxyre/Desktop/', 'joint.pickle'), "wb")
f.write(pickle.dumps(joint))
f.close()

for i in range(len(joint)):
    print('class %i: ' %i)
    for j in range(len(joint[i])):
        print('frame %i: ' %j)
        print(joint[i][j])
    print()