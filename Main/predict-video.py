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
num_model = 45
count = len(glob('D:/user/Documents/Skripsi/Output/*')) + 1
model_path = 'D:/user/Documents/Skripsi/Model/'
temp_path = 'D:/user/Documents/Skripsi/Input/Temp/'
model_file = 'modelActivity%02i.h5' % num_model
label_file = 'lb%02i.pickle' % num_model
input_path = 'D:/user/Documents/Skripsi/Input/'
input_skeleton = sys.argv[1]
output_path = 'D:/user/Documents/Skripsi/Output/'
output_video = 'Output%02i.avi' % count
size = 128
connecting_joint = [2, 1, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 8, 8, 12, 12]

# load the trained model and label from disk
print('[INFO] loading model and label ...')
print(os.path.join(model_path, model_file))
model = load_model(os.path.join(model_path, model_file), compile=False)
lb = pickle.loads(open(os.path.join(model_path, label_file), "rb").read())

#class that been used
name_class = pd.read_csv('D:/user/Documents/Skripsi/Dataset/class_name_new.csv')

y_true = []
activity = dict()
for i in range(name_class.shape[0]):
	activity[name_class['name'][i]] = 0
	if input_skeleton.split('.')[0][-2:] == name_class['code'][i][-2:]:
		y_true.append(name_class['name'][i])

# load skeleton
print('[INFO] load skeleton ...')
#read skeleton
if input_skeleton.split('.')[1] == 'csv':
	df = pd.read_csv(os.path.join(input_path, input_skeleton))
	bodyinfo = read_csv_file(df)
	connecting_joint = [2, 1, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 8, 8, 12, 12]
	height = 1080
	width = 1920
	#make blank images
	frame = np.zeros(shape=[height, width, 3], dtype=np.uint8)
	color = tuple(reversed([0,0,0]))
	frame[:] = color
	for i in range(len(bodyinfo)): 
		for j in range(25):
			try:
				# red for line
				rv = 255
				gv = 0
				bv = 0
				#search for joint that connect
				k = connecting_joint[j] - 1 
				#get joint x and y
				joint = bodyinfo[i]['joints'][j]
			
				dx = np.int32(round(float(joint['x'] * 4.016 * 100 + width/2)))
				dy = np.int32(round(float( - joint['y'] * 4.016 * 100 + height/2)))
				joint2 = bodyinfo[i]['joints'][k]
				dx2 = np.int32(round(float(joint2['x'] * 4.016 * 100 + width/2)))
				dy2 = np.int32(round(float( - joint2['y'] * 4.016 * 100 + height/2)))
				#get pixel for the line 
				line = bresenham_line(dx, dy, dx2, dy2)
				#write line per pixel
				for l in range(len(line)):
					dx = line[l][0]
					dy = line[l][1]
					frame = cv2.circle(frame, (dx, dy), radius=2, color=(bv, gv, rv), thickness=-1)

				#green color for points/ joints
				rv = 0
				gv = 255
				bv = 0
				#get x and y
				joint = bodyinfo[i]['joints'][j]
				dx = np.int32(round(float(joint['x'] * 4.016 * 100 + width/2)))
				dy = np.int32(round(float( - joint['y'] * 4.016 * 100 + height/2)))
				#write joint
				frame = cv2.circle(frame, (dx, dy), radius=5, color=(bv, gv, rv), thickness=-1)
			except:
				#if theres error then break
				check = True
				break
		#save file
		filename = os.path.join(temp_path, "frame%i.jpg" % i)
		# frame = cv2.resize(frame, (224, 224))
		cv2.imwrite(filename, frame)
elif input_skeleton.split('.')[1] == 'skeleton':
	bodyinfo = read_skeleton_file(os.path.join(input_path, input_skeleton))
	for i in range(len(bodyinfo)):
		frame = np.zeros(shape=[1080, 1920, 3], dtype=np.uint8)
		color = tuple(reversed([0,0,0]))
		frame[:] = color
		for j in range(25):
			# red for line
			rv = 255
			gv = 0
			bv = 0
			#search for joint that connect
			k = connecting_joint[j] - 1
			#get joint x and y
			joint = bodyinfo[i][0]['joints'][j]
			dx = np.int32(round(float(joint['colorX'])))
			dy = np.int32(round(float(joint['colorY'])))
			joint2 = bodyinfo[i][0]['joints'][k]
			dx2 = np.int32(round(float(joint2['colorX'])))
			dy2 = np.int32(round(float(joint2['colorY'])))
			#get pixel for the line 
			line = bresenham_line(dx, dy, dx2, dy2)
			#write line per pixel
			for l in range(len(line)):
				dx = line[l][0]
				dy = line[l][1]
				frame = cv2.circle(frame, (dx, dy), radius=2, color=(bv, gv, rv), thickness=-1)

			#green color for points/ joints
			rv = 0
			gv = 255
			bv = 0
			#get x and y
			joint = bodyinfo[i][0]['joints'][j]
			dx = np.int32(round(float(joint['colorX'])))
			dy = np.int32(round(float(joint['colorY'])))
			#write joint
			frame = cv2.circle(frame, (dx, dy), radius=5, color=(bv, gv, rv), thickness=-1)		
		#save file
		filename = os.path.join(temp_path, "frame%i.jpg" % i)
		# frame = cv2.resize(frame, (224, 224))
		cv2.imwrite(filename, frame)

writer = None
(W, H) = (None, None)

images = glob(os.path.join(temp_path, '*'))

# if num_model < 21 and num_model > 1 :
	
# 	Q = deque(maxlen=size)
# 	# loop over frames from the video file stream
# 	print('[INFO] loop over frames ...')
# 	for i in range(len(images)):
# 		# read the next frame from the file
# 		frame = cv2.imread(images[i])

# 		# if the frame dimensions are empty, grab them
# 		if W is None or H is None:
# 			(H, W) = frame.shape[:2]

# 		# clone the output frame, then convert it from BGR to RGB
# 		# ordering, resize the frame to a fixed 224x224
# 		output = frame.copy()
# 		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# 		frame = cv2.resize(frame, (224, 224)).astype('float32')

# 		# make predictions on the frame and then update the predictions
# 		# queue
# 		preds = model.predict(np.expand_dims(frame, axis=0))[0]
# 		Q.append(preds)

# 		# perform prediction averaging over the current history of
# 		# previous predictions
# 		results = np.array(Q).mean(axis=0)
# 		i = np.argmax(results)
# 		label = lb.classes_[i]

# 		# draw the activity on the output frame
# 		text = 'activity: {}'.format(label)
# 		activity[label] = activity[label] + 1
# 		cv2.putText(output, text, (10, 25), cv2.FONT_HERSHEY_PLAIN,
# 			1.5, (255, 255, 255), 3)

# 		# check if the video writer is None
# 		if writer is None:
# 			# initialize our video writer
# 			fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# 			writer = cv2.VideoWriter(os.path.join(output_path, output_video), fourcc, 30,
# 				(W, H), True)

# 		# write the output frame to disk
# 		writer.write(output)

# 		# show the output image
# 		key = cv2.waitKey(1) & 0xFF

# 		# if the `q` key was pressed, break from the loop
# 		if key == ord('q'):
# 			break
# else:
count = math.floor(float(len(images) - 5) / 10)

framecount = 0
temp_image = []
test_image = []
# loop over frames from the video file stream
print('[INFO] loop over frames ...')
for i in range(len(images)):
	
	# read the next frame from the file
	frame = cv2.imread(images[i])

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# clone the output frame, then convert it from BGR to RGB
	# ordering, resize the frame to a fixed 224x224
	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (224, 224)).astype('float32')

	if (i - 5) % count == 0 :
		framecount = framecount + 1
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		frame = np.expand_dims(frame, axis=2)  
		temp_image.append(frame)
		
	# draw the activity on the output frame
	cv2.putText(output, text, (10, 25), cv2.FONT_HERSHEY_PLAIN,
		1.5, (255, 255, 255), 3)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		writer = cv2.VideoWriter(os.path.join(output_path, output_video), fourcc, 30,
			(W, H), True)

	# write the output frame to disk
	writer.write(output)

	# show the output image
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord('q'):
		break

preds = model.predict(np.concatenate(temp_image, axis=2))[0]
label = lb.classes_[preds]
activity[label] = activity[label] + 1



tn, fp, fn, tp = confusion_matrix(y_true, max(activity, key=activity.get)).ravel()
f = open(os.path.join(output_path, 'report%s.txt' %input_skeleton.split('.')[0]), 'w')
f.write('Actual: %s\n' %y_true)
f.write('Predict: %s\n' %max(activity, key=activity.get))
f.write('TN: %i\n' %tn)
f.write('FP: %i\n' %fp)
f.write('FN: %i\n' %fn)
f.write('TP: %i\n' %tp)
f.close()

#result
print(activity)
print('[INFO] RESULT ...')
print('ACTIVITY: ' + max(activity, key=activity.get))


# release the file pointers
print('[INFO] cleaning up...')
writer.release()
# for f in images:
#     os.remove(f)