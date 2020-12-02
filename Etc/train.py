import os
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, Flatten, MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.optimizers import Adam
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
from glob import glob

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

# #initialize
# num_train = 0 #25
# learn_rate = 1e-5 
# num_epochs = 40 #25
# batchsize = 16
# drop_out = 0.2 #0.4

# #file to save
# weight_final = 'modelActivity%02i.h5' % num_train
# lb_file = 'lb%02i.pickle' % num_train
# loss_file = 'lossplot%02i.png' % num_train
# acc_file = 'accplot%02i.png' % num_train
# summary_file = 'report%02i.txt' % num_train
# configure_file = 'config%02i.txt' % num_train
# acc_n_loss_file = 'history%02i.csv' % num_train
# classification_report_file = 'classification%02i.csv' % num_train

# #path
# data_path = 'C:/train/'
# model_path = 'D:/user/Documents/Skripsi/Model/'
# report_path = 'D:/user/Documents/Skripsi/Hasil Tes/'
# check_path = 'D:/user/Documents/Skripsi/checkpoint/'

# #save configuration
# f = open(os.path.join(report_path, configure_file), 'w')
# f.write('Learning rate : %f\n' % learn_rate)
# f.write('Epoch : %i\n' % num_epochs)
# f.write('Batchsize : %i\n' % batchsize)
# f.write('Drop Out : %f\n' % drop_out)
# f.write('Start time: %s\n' % datetime.datetime.now() )
# f.close()

print("[INFO] load image ...")
#load pickle of image and label
images = glob('C:/train_tests_crop/*')

np_img = []
for i in range(2):
    img = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
        
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
    frame = np.expand_dims(frame, axis=2)  
    np_img.append(frame)




# converting the list of image to numpy array
X = np.array(np_img)

# converting the list of label to numpy array
y = np.array(['drink', 'eat'])

# perform one-hot encoding on the labels
lb = LabelBinarizer()
y = lb.fit_transform(y)


train_aug = ImageDataGenerator(rescale=1./255)
train_val = train_aug.flow(X, y, batch_size=16)
# #split data
# print("[INFO] splitting data ...")
# trainX, testX, trainY, testY = train_test_split(X, y, random_state = 42, test_size = 0.20, stratify = y)

# # #release memory
# del X
# del y
# gc.collect()
# del gc.garbage[:] 

#load VGG16 network
print("[INFO] load vgg16 model ...")
baseModel = Sequential()
baseModel.add(MaxPooling2D(pool_size=(2,2), strides=2, input_shape=(224,224,1)))

# # add callbacks for model
# print("[INFO] adding callbacks ...")
# time_callbacks = TimeHistory()
# model_callbacks =[
#     #for earlystoping
#     # EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, mode='auto'),
#     #for check point
#     ModelCheckpoint(filepath=os.path.join(check_path, 'model.{epoch:02d}-{val_loss:.2f}.h5'), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min'),
#     #for record time
#     time_callbacks
# ] 

# fullly connected layer configuration
# print("[INFO] configure fully connected layer ...")
# headModel = baseModel.output
# # headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
# headModel = Flatten(input_shape=baseModel.output_shape[1:])(headModel)
# headModel = Dense(512, activation='relu')(headModel)
# headModel = Dropout(drop_out)(headModel) #coba
# headModel = Dense(len(lb.classes_), activation='softmax')(headModel)

# # setting up model
# print("[INFO] setting up model ...")
# model = Model(inputs=baseModel.input, outputs=headModel)

# freeze base model trainable parameters
# for layer in baseModel.layers:
#     layer.trainable = False

# # writing summary
# print("[INFO] writing summary ...")
# with open(os.path.join(report_path, summary_file),'w') as fh:
#     # Pass the file handle in as a lambda function to make it callable
#     model.summary(print_fn=lambda x: fh.write(x + '\n'))

# compile model
print("[INFO] compiling ...")
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# release memory again
gc.collect()
del gc.garbage[:] 

#train the head of the network for a few epochs (all other layers are frozen) -- this will allow the new FC layers to start to become initialized with actual "learned" values versus pure random
print("[INFO] training ...")
H = model.fit(
    x=X,
    y=y,
    batch_size=16,
    validation_split=0.10,
    shuffle=True,
    epochs=1
    )

f = open('C:/Users/cxyre/Desktop/file.txt'), 'a')
f.write('Finish time: %s\n' % datetime.datetime.now())
f.close()

# # evaluate the network
# print("[INFO] evaluating ...")
# predictions = model.predict(x=testX.astype('float32'), batch_size=batchsize)
# report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_, output_dict=True)
# print("classification report"),
# print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))
# df = pd.DataFrame(report).transpose()
# df.to_csv(os.path.join(report_path, classification_report_file), index = False)
# scores = model.evaluate(testX, testY, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
# f = open(os.path.join(report_path, configure_file), 'a')
# f.write("%s: %.2f%%\n" % (model.metrics_names[0], scores[0]*100))
# f.close()


# # plot the training loss and accuracy
# print("[INFO] making plot for loss and accuracy...")
# #loss
# plt.style.use('ggplot')
# plt.figure()
# plt.plot(H.history["loss"], label="train_loss")
# plt.plot(H.history["val_loss"], label="val_loss")
# plt.title("Training Loss on Dataset")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss")
# plt.legend(loc="lower left")
# plt.savefig(os.path.join(report_path, loss_file))
# #accuracy
# plt.style.use('ggplot')
# plt.figure()
# plt.plot(H.history["accuracy"], label="train_acc")
# plt.plot(H.history["val_accuracy"], label="val_acc")
# plt.title("Training Accuracy on Dataset")
# plt.xlabel("Epoch #")
# plt.ylabel("Accuracy")
# plt.legend(loc="lower left")
# plt.savefig(os.path.join(report_path, acc_file))
# #save history
# df = pd.DataFrame()
# df['accuracy'] = H.history['accuracy']
# df['val_accuracy'] = H.history['val_accuracy']
# df['loss'] = H.history['loss']
# df['val_loss'] = H.history['val_loss']
# df['time'] = time_callbacks.times
# df.to_csv(os.path.join(report_path, acc_n_loss_file))

# # saving weight
# print("[INFO] saving weight ...")
# model.save(os.path.join(model_path, weight_final))

# # saving the label binarizer to disk
# print("[INFO] saving label ...")
# f = open(os.path.join(model_path, lb_file), "wb")
# f.write(pickle.dumps(lb))
# f.close()

# # release memory
# print("[INFO] Done ...")
# gc.collect()
# del gc.garbage[:] 
# # save finish time
# f = open(os.path.join(report_path, configure_file), 'a')
# f.write('Finish time: %s\n' % datetime.datetime.now())
# f.close()