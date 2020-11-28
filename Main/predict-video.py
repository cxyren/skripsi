from keras.models import load_model
from collections import deque
import numpy as np
import pickle
import cv2
import os

#initialize
num_model = 1
count = 1
model_path = 'D:/user/Documents/Skripsi/Model/'
model_file = 'modelActivity%02i.model' % num_model
label_file = 'lb%02i.pickle' % num_model
input_path = 'D:/user/Documents/Skripsi/Input/'
input_skeleton = ''
output_path = 'D:/user/Documents/Skripsi/Output/'
output_video = 'Output%02i.avi' % count
size = 128

# load the trained model and label from disk
print("[INFO] loading model and label ...")
model = load_model(os.path.join(model_path, model_file))
lb = pickle.loads(open(os.path.join(model_path, label_file), "rb").read())

Q = deque(maxlen=size)

# load video
print("[INFO] load skeleton ...")
videocap = cv2.VideoCapture(os.path.join(input_path, input_video))
writer = None
(W, H) = (None, None)

# loop over frames from the video file stream
print("[INFO] loop over frames ...")
while True:
	# read the next frame from the file
	(ret, frame) = videocap.read()

	# if the frame was not grabbedm then break
	if not ret:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# clone the output frame, then convert it from BGR to RGB
	# ordering, resize the frame to a fixed 224x224, and then
	# perform mean subtraction
	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (224, 224)).astype("float32")

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
	text = "activity: {}".format(label)
	cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
		1.25, (0, 255, 0), 5)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(os.path.join(output_path, output_video), fourcc, 30,
			(W, H), True)

	# write the output frame to disk
	writer.write(output)

	# show the output image
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
videocap.release()