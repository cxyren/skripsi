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
# import tensorflow as tf

# tf.config.experimental.set_visible_devices([], 'GPU')
# # tf.device('/cpu:0')

# physical_devices = tf.config.list_physical_devices('CPU')
# try:
#     # Disable first GPU
#     tf.config.set_visible_devices(physical_devices, 'CPU')
#     logical_devices = tf.config.list_logical_devices('CPU')
#     # Logical device was not created for first GPU
#     assert len(logical_devices) == len(physical_devices) - 1
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass

#initialize
learn_rate = 1e-3
num_epochs = 25 #pengujian
batchsize = 8
drop_out = 0.2 #pengujian juga kalo bisa
weight_final = 'modelActivity03.h5'
lb_file = 'lb03.pickle'
loss_file = 'lossplot03.png'
acc_file = 'accplot03.png'
summary_file = 'report03.txt'
classification_report_file = 'classification03.txt'

#training data
train = pd.read_csv('D:/user/Documents/Skripsi/Dataset/fix/train_newest2.csv')

#path
image_path =  'C:/train_image2/' #'D:/user/Documents/Skripsi/Dataset/train/'
model_path = 'D:/user/Documents/Skripsi/Model/'
report_path = 'D:/user/Documents/Skripsi/Hasil Tes/'
check_path = 'D:/user/Documents/Skripsi/checkpoint/'

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
    img = cv2.resize(img, (150, 150))
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

print("[INFO] initialize training data augmentation ...")
#initialize the training data augmentation object
train_aug = ImageDataGenerator(
    rotation_range = 30, 
    width_shift_range = 0.2, 
    height_shift_range = 0.2, 
    shear_range = 0.15, 
    zoom_range = 0.15, 
    fill_mode = 'nearest' 
    )
train_batches = train_aug.flow(trainX, trainY, batch_size=batchsize)

# initialize the validation/testing data augmentation
val_aug = ImageDataGenerator()
val_batches = val_aug.flow(testX, testY, batch_size=batchsize)

print("[INFO] load vgg16 model ...")
#load VGG16 network
baseModel = VGG16(weights='imagenet',include_top=False, input_shape=(150, 150, 3))

# print("[INFO] adding callbacks ...")
# # add callbacks for model
# model_callbacks =[
#     #for earlystoping
#     EarlyStopping(monitor='val_accuracy', min_delta=0, patience=40, verbose=1, mode='auto'),
#     #for check point
#     ModelCheckpoint(filepath=os.path.join(check_path, 'model.{epoch:02d}-{val_loss:.2f}.h5'), monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
# ] 

print("[INFO] configure fully connected layer ...")
# fullly connected layer configuration
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
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
with open(os.path.join(report_path, summary_file),'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

print("[INFO] compiling ...")
# compile model
model.compile(optimizer=Adam(learning_rate=learn_rate), loss='categorical_crossentropy', metrics=['accuracy'])

gc.collect()
del gc.garbage[:] 

print("[INFO] training ...")
#train the head of the network for a few epochs (all other layers are frozen) -- this will allow the new FC layers to start to become initialized with actual "learned" values versus pure random
H = model.fit(
    x=train_aug, 
    batch_size=batchsize,
    steps_per_epoch=len(trainX) // batchsize,
    validation_data=val_aug,
    validation_steps=len(testX) // batchsize,
    epochs=num_epochs
    )

print("[INFO] evaluating ...")
# evaluate the network
predictions = model.predict(x=testX.astype('float32'), batch_size=batchsize)
report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_)
print("classification report"),
print(report)
df = pd.DataFrame(report).transpose()
df.to_csv(os.path.join())
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