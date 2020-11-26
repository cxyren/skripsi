import os
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Flatten, AveragePooling2D
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
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
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		tf.config.experimental.set_memory_growth(gpus[0], True)
		#tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7168)])
	except RuntimeError as e:
		print(e)

def classification_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(os.path.join(report_path, classification_report_file), index = False)

#initialize
num_train = 4
learn_rate = 1e-3
num_epochs = 1 #pengujian
batchsize = 8
drop_out = 0.4 #pengujian juga kalo bisa

# weight_previous= 'modelActivity%02i.h5' % (num_train - 1)
weight_final = 'modelActivity%02i.h5' % num_train
lb_file = 'lb%02i.pickle' % num_train
loss_file = 'lossplot%02i.png' % num_train
acc_file = 'accplot%02i.png' % num_train
summary_file = 'report%02i.txt' % num_train
configure_file = 'config%02i.txt' % num_train
classification_report_file = 'classification%02i.csv' % num_train

#training data
train = pd.read_csv('D:/user/Documents/Skripsi/Dataset/fix/train_newest3.csv')

#path
image_path =  'C:/train_image2/' #'D:/user/Documents/Skripsi/Dataset/train/'
model_path = 'D:/user/Documents/Skripsi/Model/'
report_path = 'D:/user/Documents/Skripsi/Hasil Tes/'
check_path = 'D:/user/Documents/Skripsi/checkpoint/'

f = open(os.path.join(report_path, configure_file), 'w')
f.write('Learning rate : %f\n' % learn_rate)
f.write('Epoch : %i\n' % num_epochs)
f.write('Batchsize : %i\n' % batchsize)
f.write('Drop Out : %f\n' % drop_out)
f.write('Start time: %s\n' % datetime.datetime.now() )
f.close()

# creating an empty list
train_image = []
label = []

print("[INFO] load image ...")
for i in tqdm(range(train.shape[0])):
    if not train['class'][i]:
        continue
    # loading the image and keeping the target size as (224,224,3)
    img = cv2.imread(os.path.join(image_path, train['image'][i]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    # appending the image to the train_image list
    train_image.append(img)
    label.append(train['class'][i])
    del img 
del train

# converting the list to numpy array
X = np.array(train_image)

# separating the target
y = np.array(label)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
y = lb.fit_transform(y)

print("[INFO] Splitting data ...")
# creating the training and validation set
trainX, testX, trainY, testY = train_test_split(X, y, random_state=42, test_size=0.25, stratify = y)

#release memory
del X
del y
gc.collect()
del gc.garbage[:] 

print("[INFO] load vgg16 model ...")
#load VGG16 network
baseModel = VGG16(weights='imagenet',include_top=False, input_shape=(224, 224, 3))

print("[INFO] adding callbacks ...")
# add callbacks for model
model_callbacks =[
    #for earlystoping
    EarlyStopping(monitor='val_accuracy', min_delta=0, patience=70, verbose=1, mode='auto'),
    #for check point
    ModelCheckpoint(filepath=os.path.join(check_path, 'model.{epoch:02d}-{val_loss:.2f}.h5'), monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
] 

print("[INFO] configure fully connected layer ...")
# fullly connected layer configuration
headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(input_shape=baseModel.output_shape[1:])(headModel)
headModel = Dense(512, activation='relu')(headModel)
headModel = Dropout(drop_out)(headModel) #coba
headModel = Dense(len(lb.classes_), activation='softmax')(headModel)

print("[INFO] setting up model ...")
#setting up model
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they wont be updated during the training process
for layer in baseModel.layers:
    layer.trainable = False

print("[INFO] writing summary ...")
# Open the file
with open(os.path.join(report_path, summary_file),'a') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

print("[INFO] compiling ...")
# compile model
model.compile(optimizer=Adam(learning_rate=learn_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# print("[INFO] load previous weight ...")
# # compile model
# model.load_weights(os.path.join(model_path, weight_previous), by_name=True)

gc.collect()
del gc.garbage[:] 

print("[INFO] training ...")
#train the head of the network for a few epochs (all other layers are frozen) -- this will allow the new FC layers to start to become initialized with actual "learned" values versus pure random
H = model.fit(
    x=trainX,
    y=trainY,
    batch_size=batchsize,
    validation_data=(testX,testY),
    epochs=num_epochs,
    callbacks=model_callbacks
    )

print("[INFO] evaluating ...")
# evaluate the network
predictions = model.predict(x=testX.astype('float32'), batch_size=batchsize)
report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_)
print("classification report"),
print(report)
classification_report_csv(report)
score = model.evaluate(val_batches, steps=100, verbose=1)
print("Accuracy is %s " % (score[1]*100))

print("[INFO] making plot for loss and accuracy...")
# plot the training loss and accuracy
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

print("[INFO] saving weight ...")
model.save(os.path.join(model_path, weight_final))

print("[INFO] saving label ...")
# serialize the label binarizer to disk
f = open(os.path.join(model_path, lb_file), "wb")
f.write(pickle.dumps(lb))
f.close()

print("[INFO] Done ...")
gc.collect()
del gc.garbage[:] 

f = open(os.path.join(report_path, configure_file), 'a')
f.write('Finish time: %s\n' % datetime.datetime.now())
f.close()