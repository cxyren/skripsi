import os
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Flatten, AveragePooling2D
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
num_train = 10
learn_rate = 1e-3
num_epochs = 25 #pengujian
batchsize = 12
drop_out = 0.3 #pengujian juga kalo bisa

#file to save
weight_final = 'modelActivity%02i.h5' % num_train
lb_file = 'lb%02i.pickle' % num_train
loss_file = 'lossplot%02i.png' % num_train
acc_file = 'accplot%02i.png' % num_train
summary_file = 'report%02i.txt' % num_train
configure_file = 'config%02i.txt' % num_train
acc_n_loss_file = 'history%02i.csv' % num_train
classification_report_file = 'classification%02i.csv' % num_train

#training data
train = pd.read_csv('D:/user/Documents/Skripsi/Dataset/fix/train_newest4.csv')

#path
image_path =  'C:/train_image/' 
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

# creating empty list
train_image = []
label = []

#load image
print("[INFO] load image ...")
for i in tqdm(range(train.shape[0])):
    if not train['class'][i]:
        continue
    # loading the image and resize to 224x224 and rgb
    img = cv2.imread(os.path.join(image_path, train['image'][i]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    # appending the image and the label into the list
    train_image.append(img)
    label.append(train['class'][i])
    del img 
del train

# converting the list of image to numpy array
X = np.array(train_image)

# converting the list of label to numpy array
y = np.array(label)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
y = lb.fit_transform(y)

# # splitting the training and validation data
# print("[INFO] Splitting data ...")
# trainX, testX, trainY, testY = train_test_split(X, y, random_state=42, test_size=0.25, stratify = y)

# #release memory
# del X
# del y
gc.collect()
del gc.garbage[:] 

#load VGG16 network
print("[INFO] load vgg16 model ...")
baseModel = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# add callbacks for model
print("[INFO] adding callbacks ...")
time_callbacks = TimeHistory()
model_callbacks =[
    #for earlystoping
    EarlyStopping(monitor='val_accuracy', min_delta=0, patience=100, verbose=1, mode='auto'),
    #for check point
    ModelCheckpoint(filepath=os.path.join(check_path, 'model.{epoch:02d}-{val_loss:.2f}.h5'), monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto'),
    #for record time
    time_callbacks
] 

# fullly connected layer configuration
print("[INFO] configure fully connected layer ...")
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(input_shape=baseModel.output_shape[1:])(headModel)
headModel = Dense(512, activation='relu')(headModel)
headModel = Dropout(drop_out)(headModel) #coba
headModel = Dense(len(lb.classes_), activation='softmax')(headModel)

# setting up model
print("[INFO] setting up model ...")
model = Model(inputs=baseModel.input, outputs=headModel)

# freeze base model trainable parameters
for layer in baseModel.layers:
    layer.trainable = False

# writing summary
print("[INFO] writing summary ...")
with open(os.path.join(report_path, summary_file),'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

# compile model
print("[INFO] compiling ...")
model.compile(optimizer=Adam(learning_rate=learn_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# release memory again
gc.collect()
del gc.garbage[:] 

#train the head of the network for a few epochs (all other layers are frozen) -- this will allow the new FC layers to start to become initialized with actual "learned" values versus pure random
print("[INFO] training ...")
H = model.fit(
    x=X,
    y=y,
    batch_size=batchsize,
    validation_split=0.25,
    epochs=num_epochs,
    callbacks=model_callbacks
    )

# evaluate the network
print("[INFO] evaluating ...")
predictions = model.predict(x=X.astype('float32'), batch_size=batchsize)
report = classification_report(y.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_, output_dict=True)
print("classification report"),
print(report)
df = pd.DataFrame(report).transpose()
df.to_csv(os.path.join(report_path, classification_report_file), index = False)

# plot the training loss and accuracy
print("[INFO] making plot for loss and accuracy...")
#loss
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, num_epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, num_epochs), H.history["val_loss"], label="val_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(os.path.join(report_path, loss_file))
#accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, num_epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, num_epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig(os.path.join(report_path, acc_file))
#save history
df = pd.DataFrame()
df['accuracy'] = H.history['accuracy']
df['val_accuracy'] = H.history['accuracy']
df['loss'] = H.history['loss']
df['val_loss'] = H.history['val_loss']
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