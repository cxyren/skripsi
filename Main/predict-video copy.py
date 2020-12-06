from keras.models import load_model
from collections import deque
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np
import math
import pickle
import cv2
import os
import sys

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

def read_csv_file(df):
	framecount = df['EventIndex'].iloc[-1] + 1
	bodyinfo = []
	for i in range(int(framecount)):
		body = {
			"bodyID": df['SkeletonId'][i],
			"handLeftConfidence": df['HandLeftConfidence'][i],
			"handLeftState": df['HandLeftState'][i],
			"handRightConfidence": df['HandRightConfidence'][i],
			"handRightState": df['HandRightState'][i],
			"time": df['Time'][i],
			"joints": []
		}
		for j in range(25):
			if j == 0:
				joint={
					"x": df['PositionX'][i],
					"y": df['PositionY'][i],
					"z": df['PositionZ'][i],
				}
			else:
				joint={
					"x": df['PositionX.%i'%j][i],
					"y": df['PositionY.%i'%j][i],
					"z": df['PositionZ.%i'%j][i],
				}
			body["joints"].append(joint)
		bodyinfo.append(body)
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

#initialize
num_model = 54
count = len(glob('D:/user/Documents/Skripsi/Output/*')) + 1
model_path = 'D:/user/Documents/Skripsi/Model/'
temp_path = 'D:/user/Documents/Skripsi/Input/Temp/'
model_file = 'modelActivity%02i.h5' % num_model
label_file = 'lb%02i.pickle' % num_model
input_path = 'D:/user/Documents/Skripsi/Input/CSV/'
input_skeleton = sys.argv[1]
output_path = 'D:/user/Documents/Skripsi/Output/'
output_video = 'Output%02i.avi' % count

size = 128

connecting_joint = [2, 1, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 8, 8, 12, 12]

# load the trained model and label from disk
print('[INFO] loading model and label ...')
print(os.path.join(model_path, model_file))
model = load_model(os.path.join(model_path, model_file))
lb = pickle.loads(open(os.path.join(model_path, label_file), "rb").read())

#class that been used
name_class = pd.read_csv('D:/user/Documents/Skripsi/Dataset/class_name_new.csv')

y_true = []
activity = dict()
for i in range(name_class.shape[0]):
	activity[name_class['name'][i]] = 0
	if input_skeleton.split('.')[0][-2:] == name_class['code'][i][-2:]:
		y_true.append(name_class['name'][i])

print('Skeleton name: %s' % input_skeleton)
# load skeleton
print('[INFO] load skeleton ...')
#read skeleton
X = pickle.loads(open(os.path.join(data_path, 'new_testx.pickle'), "rb").read())
y = pickle.loads(open(os.path.join(data_path, 'new_testy.pickle'), "rb").read())

print('[INFO] Model Predicting ...')
preds = model.predict(x=testX.astype('float32'))[0]
Q.append(preds)
results = np.array(Q).mean(axis=0)
i = np.argmax(results)
label = lb.classes_[i]
activity[label] = activity[label] + 1
y_predict = []
y_predict.append(max(activity, key=activity.get))

#result
print('[INFO] RESULT ...')
print('ACTUAL: %s' % y_true)
print('ACTIVITY: %s' % y_predict)

result = confusion_matrix(y_true=y_true, y_pred=y_predict).ravel()
if len(result) == 4:
	tn, fp, fn, tp = result
else:
	tp = result
	tn = 0
	fp = 0
	fn = 0
f = open(os.path.join(output_path, 'report%s.txt' %input_skeleton.split('.')[0]), 'w')
f.write('Actual: %s\n' % y_true)
f.write('Predict: %s\n' % y_predict)
f.write('TN: %i\n' % tn)
f.write('FP: %i\n' % fp)
f.write('FN: %i\n' % fn)
f.write('TP: %i\n' % tp)
f.close()

# release the file pointers
print('[INFO] cleaning up...')
for f in images:
    os.remove(f)