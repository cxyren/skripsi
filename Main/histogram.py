from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from keras.callbacks import Callback
from glob import glob
from tqdm import tqdm
from sklearn import metrics
import pickle
import pandas as pd
import gc
import datetime
import time
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
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

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

#setting up tensorflow-gpu
gpu = tf.config.experimental.list_physical_devices('GPU')
if gpu:
	try:
		tf.config.experimental.set_memory_growth(gpu[0], True)
	except RuntimeError as e:
		print(e)

num_train = 3
learn_rate = 1e-5
num_epochs = 100 #25
batchsize = 100
test_case = 3

#file to save
weight_final = 'NEURALNETWORK%02i.h5' % num_train
lb_file = 'NEURALNETWORKLB%02i.pickle' % num_train
loss_file = 'LOSSNEURAL%02i.png' % num_train
acc_file = 'ACCNEURAL%02i.png' % num_train
summary_file = 'NEURALNETWORK%02i.txt' % num_train
acc_n_loss_file = 'HISTORYNEURAL%02i.csv' % num_train
classification_report_file = 'CLASSNEURAL%02i.csv' % num_train
configure_file = 'NEURALCONFIG%02i.txt' % num_train

#path
model_path = 'D:/user/Documents/Skripsi/Model/'
report_path = 'D:/user/Documents/Skripsi/Hasil Tes/'

#save configuration
f = open(os.path.join(report_path, configure_file), 'w')
f.write('Learning rate : %f\n' % learn_rate)
f.write('Epoch : %i\n' % num_epochs)
f.write('Batchsize : %i\n' % batchsize)
f.write('Start time: %s\n' % datetime.datetime.now() )
f.close()

#list of image
list_img = pd.read_csv('D:/user/Documents/Skripsi/Dataset/fix/train_newest9.csv')

if not os.path.isfile("C:/users/cxyre/Desktop/blockx%i.pickle" %test_case) and not os.path.isfile("C:/users/cxyre/Desktop/blocky%i.pickle" %test_case):
    name_class = pd.read_csv('D:/user/Documents/Skripsi/Dataset/class_name_new.csv')

    class_code = dict()
    for i in range(name_class.shape[0]):
        class_code[name_class['code'][i]] = name_class['name'][i]

    class_code.pop('A002')
    class_code.pop('A009')
    class_code.pop('A017')
    class_code.pop('A037')
    class_code.pop('A041')
    class_code.pop('A044')
    class_code.pop('A047')

    count = -1
  
    temp_name = dict()
    image_count = 0

    x = []
    y = []
    
    frame_count = 0
    
    #used setup
    setup_num = dict()
    setup_num['S003'] = True
    setup_num['S004'] = True
    setup_num['S005'] = True
    setup_num['S006'] = True
    setup_num['S008'] = True
    setup_num['S009'] = True

    print('[INFO] MAKING BLOCK ..')
    #making block
    for i in tqdm(range(len(list_img))):
        if list_img['image'][i].split('_')[0][-4:] not in class_code:
            continue
        
        if image_count == 0:
            temp_name[list_img['skeleton'][i]] = True
            frame_count = 0
            block = []
            for j in range(56):
                block.append([])
                for k in range(56):
                    block[j].append(0)
            x.append(block)
            count = count + 1
            y.append(class_code.get(list_img['image'][i].split('_')[0][-4:]))

        if list_img['skeleton'][i] not in temp_name:
            frame_count = 0
            temp_image.clear()
            image_count = 0
            del x[count]
            del y[count]
            count = count - 1
            continue
            
        # print(os.path.join('C:/train_tests_crop/', list_img['image'][i]))
        img = cv2.imread(os.path.join('C:/train_test2_crop/', list_img['image'][i]), cv2.IMREAD_GRAYSCALE)
        # print(img.shape)
        #making empty frame    
        frame = np.zeros(shape=[224, 224], dtype=np.uint8)
        if img.shape[1] < img.shape[0]:
            img = cv2.resize(img, (int(img.shape[1]*(224/img.shape[0])), 224))
            
            #relocating image
            center_x = frame.shape[1] / 2
            center_x2 = img.shape[1] / 2
            
            frame[int(0):int(224), int(center_x - center_x2):int(center_x + center_x2)] = img
        else:
            img = cv2.resize(img, (224, int(img.shape[0]*(224/img.shape[1]))))

            #relocating image
            center_y = frame.shape[0] / 2
            center_y2 = img.shape[0] / 2
            
            frame[int(center_y - center_y2):int(center_y + center_y2), int(0):int(224)] = img
            
        #resize image
        frame = cv2.resize(frame, (224, 224))
        # frame = np.expand_dims(frame, axis=2)  
        frame = frame.astype('float64')
        frame *= 255.0/frame.max()

        frame_count = frame_count + 1
        for j in range(frame.shape[1]):
            for k in range(frame.shape[0]):
                if x[count][math.ceil(k*1.0/4.0) - 1][math.ceil(j*1.0/4.0) - 1] == frame_count:
                    continue
                if frame[k, j] > 250:
                    x[count][math.ceil(k*1.0/4.0) - 1][math.ceil(j*1.0/4.0) - 1] = x[count][math.ceil(k*1.0/4.0) - 1][math.ceil(j*1.0/4.0) - 1] + 1

    print(x)
    print(y)
    print("[INFO] saving image data ...")
    f = open(os.path.join('C:/users/cxyre/Desktop/', 'blockx%i.pickle'%test_case), "wb")
    f.write(pickle.dumps(x))
    f.close()

    f = open(os.path.join('C:/users/cxyre/Desktop/', 'blocky%i.pickle'%test_case), "wb")
    f.write(pickle.dumps(y))
    f.close()

print("[INFO] load image ...")
#load pickle of image and label
x = pickle.loads(open(os.path.join('C:/users/cxyre/Desktop/', 'blockx1.pickle'), "rb").read())
y = pickle.loads(open(os.path.join('C:/users/cxyre/Desktop/', 'blocky1.pickle'), "rb").read())

print('[INFO] CONVERT INTO NUMPY ARRAY ..')
temp_x = []
for i in x:
  temp = np.array(i)
  temp_x.append(temp.flatten())
x = np.array(temp_x).astype('int8')
del temp_x
print(len(x))
print(len(x[0]))
y = np.array(y)
#one hot encoder
lb = LabelBinarizer()
y = lb.fit_transform(y)

print(len(y))
print(len(y[0]))
print(y[0])
print(lb.classes_)

print('[INFO] SPLITTING DATA ..')
trainX, testX, trainY, testY = train_test_split(x, y, random_state = 42, test_size = 0.2, stratify = y)

print(len(trainX))
print(len(trainX[0]))
print('[INFO] SETTING UP MODEL ..')
model = Sequential()
model.add(Dense(128, input_dim=3136, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(len(lb.classes_), activation='softmax'))


# compile the keras model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learn_rate), metrics=['accuracy'])

# writing summary
print("[INFO] writing summary ...")
with open(os.path.join(report_path, summary_file),'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

print("[INFO] adding callbacks ...")
time_callbacks = TimeHistory()
model_callbacks =[
    #for record time
    time_callbacks
] 

print("[INFO] TRAINING ...")
# fit the keras model on the dataset
H = model.fit(trainX, trainY, epochs=150, batch_size=batchsize, callbacks=model_callbacks)

# evaluate the network
print("[INFO] evaluating ...")
predictions = model.predict(x=testX.astype('float32'), batch_size=batchsize)
report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_, output_dict=True)
print("classification report"),
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))
df = pd.DataFrame(report).transpose()
df.to_csv(os.path.join(report_path, classification_report_file), index = False)
scores = model.evaluate(testX, testY, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
f = open(os.path.join(report_path, configure_file), 'a')
f.write("%s: %.2f%%\n" % (model.metrics_names[0], scores[0]))
f.write("%s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100))
f.close()


# plot the training loss and accuracy
print("[INFO] making plot for loss and accuracy...")
#loss
plt.style.use('ggplot')
plt.figure()
plt.plot(H.history["loss"], label="train_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(os.path.join(report_path, loss_file))
#accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(H.history["accuracy"], label="train_acc")
plt.title("Training Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig(os.path.join(report_path, acc_file))
#save history
df = pd.DataFrame()
df['accuracy'] = H.history['accuracy']
df['loss'] = H.history['loss']
df['time'] = time_callbacks.times
df.to_csv(os.path.join(report_path, acc_n_loss_file))

# saving weight
print("[INFO] saving weight ...")
model.save(os.path.join(model_path, weight_final))

# saving the label binarizer to disk
print("[INFO] saving label ...")
f = open(os.path.join(model_path, lb_file), "wb")
f.write(pickle.dumps(lb))
f.close()

# release memory
print("[INFO] Done ...")
gc.collect()
del gc.garbage[:] 
# save finish time
f = open(os.path.join(report_path, configure_file), 'a')
f.write('Finish time: %s\n' % datetime.datetime.now())
f.close()