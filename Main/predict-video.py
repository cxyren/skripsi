from keras.models import load_model
from collections import deque
from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np
import pickle
import cv2
import os

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
num_model = 1
count = 1
model_path = 'D:/user/Documents/Skripsi/Model/'
temp_path = 'D:/user/Documents/Skripsi/Input/Temp/'
model_file = 'modelActivity%02i.model' % num_model
label_file = 'lb%02i.pickle' % num_model
input_path = 'D:/user/Documents/Skripsi/Input/'
input_skeleton = ''
output_path = 'D:/user/Documents/Skripsi/Output/'
output_video = 'Output%02i.avi' % count
size = 128
connecting_joint = [2, 1, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 8, 8, 12, 12]

# load the trained model and label from disk
print('[INFO] loading model and label ...')
model = load_model(os.path.join(model_path, model_file))
lb = pickle.loads(open(os.path.join(model_path, label_file), "rb").read())

Q = deque(maxlen=size)

# load skeleton
print('[INFO] load skeleton ...')
#read skeleton
df = pd.read_csv(os.path.join(input_path, input_skeleton))
name_img = []
for i in tqdm(range(df.shape[0])):
	#read skeleton here
	print('ELLO')
# videocap = cv2.VideoCapture(os.path.join(input_path, input_video))
writer = None
(W, H) = (None, None)

images = glob(os.path.join(temp_path, '*'))
# loop over frames from the video file stream
print('[INFO] loop over frames ...')
for i in range(len(images)):
	# read the next frame from the file
	frame = cv2.imread(images[i])

	# if the frame was not grabbedm then break
	# if not ret:
	# 	break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# clone the output frame, then convert it from BGR to RGB
	# ordering, resize the frame to a fixed 224x224
	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (224, 224)).astype('float32')

	# make predictions on the frame and then update the predictions
	# queue
	preds = model.predict(np.expand_dims(frame, axis=0))[0]
	Q.append(preds)

	# perform prediction averaging over the current history of
	# previous predictions
	results = np.array(Q).mean(axis=0)
	i = np.argmax(results)
	label = lb.classes_[i]

	# draw the activity on the output frame
	text = 'activity: {}'.format(label)
	print('activity: ')
	cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
		1.25, (0, 255, 0), 5)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		writer = cv2.VideoWriter(os.path.join(output_path, output_video), fourcc, 30,
			(W, H), True)

	# write the output frame to disk
	writer.write(output)

	# show the output image
	# cv2.imshow('Output', output)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord('q'):
		break

# release the file pointers
print('[INFO] cleaning up...')
writer.release()