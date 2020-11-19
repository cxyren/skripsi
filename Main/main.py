import os
import keras
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Flatten, AveragePooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras_adabound import AdaBound
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import metrics
import pickle

#initialize
num_epochs = 10
num_split = 0
batchsize = 8
weight_final = 'modelActivity01.model'
lb_file = 'lb.pickle'
plt_file = 'plot.png'

#training
train = pd.read_csv('D:/user/Documents/Skripsi/Dataset/train_new.csv')

#path
image_path = 'D:/user/Documents/Skripsi/Dataset/train/'
model_path = 'D:/user/Documents/Skripsi/Model/'
plt_path = 'D:/user/Documents/Skripsi/github-program/main/Result/'

# creating an empty list
train_image = []
label = []

for i in tqdm(range(train.shape[0])):
    if not train['class'][i]:
        continue
    # loading the image and keeping the target size as (224,224,3)
    img = cv2.imread(os.path.join(image_path, train['image'][i]))
    # converting it to array
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    # appending the image to the train_image list
    train_image.append(img)
    label.append(train['class'][i])
    
del train

# converting the list to numpy array
X = np.array(train_image)

# separating the target
y = np.array(label)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
y = lb.fit_transform(y)

# creating the training and validation set
trainX, testX, trainY, testY = train_test_split(X, y, random_state=42, test_size=0.25, stratify = y)

del X
del y

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

#initialize the validation/testing data augmentation
val_aug = ImageDataGenerator()
val_batches = val_aug.flow(testX, testY, batch_size=batchsize)

#load ResNet-50 network
baseModel = ResNet50(
    weights='imagenet',
    include_top=False, 
    input_tensor=Input(shape=(224,224,3))
    )

# construct the head of model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(512, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation='softmax')(headModel)

#place head model on top of base model
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they wont be updated during the training process
for layer in baseModel.layers:
    layer.trainable = False

model.summary()

# compile our model (this needs to be done after our setting our layers to being non-trainable)
print("[INFO] compiling model...")
model.compile(optimizer=AdaBound(lr=1e-3, final_lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])


#train the head of the network for a few epochs (all other layers are frozen) -- this will allow the new FC layers to start to become initialized with actual "learned" values versus pure random
print("[INFO] training head...")
H = model.fit(
    x=train_batches, 
    steps_per_epoch=len(trainX) // batchsize,
    validation_data=val_batches,
    validation_steps=len(testX) // batchsize,
    epochs=num_epochs
    )

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX.astype('float32'), batch_size=batchsize)
print("confusion Metrix"),
print(confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1)))
print("classification report"),
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))
score = model.evaluate(val_batches, steps=100, verbose=1)
print("Accuracy is %s " % (score[1]*100))

# plot the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, num_epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, num_epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, num_epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, num_epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(os.path.join(plt_path, plt_file))

print("[INFO] serializing network...")
model.save(os.path.join(model_path, weight_final), save_format="h5")

# serialize the label binarizer to disk
f = open(os.path.join(model_path, lb_file), "wb")
f.write(pickle.dumps(lb))
f.close()