import os
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, Flatten, MaxPooling2D, Conv2D
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.optimizers import Adam
import keras
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics
import pickle
import gc
import datetime
import time
import tensorflow as tf

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

#initialize
num_train = 43  #25
learn_rate = 1e-4
num_epochs = 100 #25
batchsize = 16
drop_out = 0 #0.4

#file to save
weight_final = 'modelActivity%02i.h5' % num_train
lb_file = 'lb%02i.pickle' % num_train
loss_file = 'lossplot%02i.png' % num_train
acc_file = 'accplot%02i.png' % num_train
summary_file = 'report%02i.txt' % num_train
configure_file = 'config%02i.txt' % num_train
acc_n_loss_file = 'history%02i.csv' % num_train
classification_report_file = 'classification%02i.csv' % num_train

#path
data_path = 'C:/train/'
model_path = 'D:/user/Documents/Skripsi/Model/'
report_path = 'D:/user/Documents/Skripsi/Hasil Tes/'
check_path = 'D:/user/Documents/Skripsi/checkpoint/'

#save configuration
f = open(os.path.join(report_path, configure_file), 'w')
f.write('Learning rate : %f\n' % learn_rate)
f.write('Epoch : %i\n' % num_epochs)
f.write('Batchsize : %i\n' % batchsize)
f.write('Drop Out : %f\n' % drop_out)
f.write('Start time: %s\n' % datetime.datetime.now() )
f.close()

print("[INFO] load image ...")
#load pickle of image and label
X = pickle.loads(open(os.path.join(data_path, 'x4.pickle'), "rb").read())
y = pickle.loads(open(os.path.join(data_path, 'y4.pickle'), "rb").read())

# converting the list of image to numpy array
X = np.array(X)

# converting the list of label to numpy array
y = np.array(y)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
y = lb.fit_transform(y)

# #split data
print("[INFO] splitting data ...")
trainX, testX, trainY, testY = train_test_split(X, y, random_state = 42, test_size = 0.2, stratify = y)
trainX, valX, trainY, valY = train_test_split(trainX, trainY, random_state = 42, test_size = 0.1, stratify = trainY)

# #release memory
del X
del y
gc.collect()
del gc.garbage[:] 

newModel = Sequential()
# Model 1
newModel.add(Conv2D(filters=32, kernel_size=3, input_shape=(224,224,10), activation='relu')) #32
newModel.add(MaxPooling2D(pool_size=(3,3), strides=2))
newModel.add(Conv2D(filters=32, kernel_size=3, activation='relu'))#64 stride 1
newModel.add(MaxPooling2D(pool_size=(3,3), strides=2))
newModel.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
newModel.add(MaxPooling2D(pool_size=(3,3), strides=2))
newModel.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
# newModel.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
# newModel.add(Dropout(0.2))
newModel.add(Flatten())
newModel.add(Dense(128, activation='relu'))
# newModel.add(Dropout(0.2))
newModel.add(Dense(len(lb.classes_), activation='softmax'))

# Model 2
# newModel.add(Conv2D(filters=32, kernel_size=7, strides=(3,3), activation='relu', input_shape=(224,224,30)))
# newModel.add(Conv2D(filters=64, kernel_size=5, activation='relu'))
# newModel.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
# newModel.add(Dropout(0.1))
# newModel.add(Flatten())
# newModel.add(Dense(128, activation='relu'))
# newModel.add(Dropout(0.1))
# newModel.add(Dense(len(lb.classes_), activation='softmax'))


# # add callbacks for model
print("[INFO] adding callbacks ...")
time_callbacks = TimeHistory()
model_callbacks =[
    #for earlystoping
    EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, mode='auto'),
    #for check point
    ModelCheckpoint(filepath=os.path.join(check_path, 'model.{epoch:02d}-{val_loss:.2f}.h5'), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto'),
    #for record time
    time_callbacks
] 

# writing summary
print("[INFO] writing summary ...")
with open(os.path.join(report_path, summary_file),'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    newModel.summary(print_fn=lambda x: fh.write(x + '\n'))

# # compile model
print("[INFO] compiling ...")
newModel.compile(optimizer=Adam(learning_rate=learn_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# release memory again
gc.collect()
del gc.garbage[:] 

# #train the head of the network for a few epochs (all other layers are frozen) -- this will allow the new FC layers to start to become initialized with actual "learned" values versus pure random
print("[INFO] training ...")
H = newModel.fit(
    x=trainX,
    y=trainY,
    batch_size=batchsize,
    validation_data=(valX, valY),
    epochs=num_epochs,
    callbacks=model_callbacks
    )

# evaluate the network
print("[INFO] evaluating ...")
predictions = newModel.predict(x=testX.astype('float32'), batch_size=batchsize)
report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_, output_dict=True)
print("classification report"),
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))
df = pd.DataFrame(report).transpose()
df.to_csv(os.path.join(report_path, classification_report_file), index = False)
scores = newModel.evaluate(testX, testY, verbose=0)
print("%s: %.2f%%" % (newModel.metrics_names[0], scores[0]))
print("%s: %.2f%%" % (newModel.metrics_names[1], scores[1]*100))
f = open(os.path.join(report_path, configure_file), 'a')
f.write("%s: %.2f%%\n" % (newModel.metrics_names[0], scores[0]))
f.write("%s: %.2f%%\n" % (newModel.metrics_names[1], scores[1]*100))
f.close()


# plot the training loss and accuracy
print("[INFO] making plot for loss and accuracy...")
#loss
plt.style.use('ggplot')
plt.figure()
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["val_loss"], label="val_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(os.path.join(report_path, loss_file))
#accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(H.history["accuracy"], label="train_acc")
plt.plot(H.history["val_accuracy"], label="val_acc")
plt.title("Training Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig(os.path.join(report_path, acc_file))
#save history
df = pd.DataFrame()
df['accuracy'] = H.history['accuracy']
df['val_accuracy'] = H.history['val_accuracy']
df['loss'] = H.history['loss']
df['val_loss'] = H.history['val_loss']
df['time'] = time_callbacks.times
df.to_csv(os.path.join(report_path, acc_n_loss_file))

# saving weight
print("[INFO] saving weight ...")
newModel.save(os.path.join(model_path, weight_final))

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