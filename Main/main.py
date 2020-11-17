import os
import keras
from keras.models import Model
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras_adabound import Adabound
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import metrics

#initialize
num_epoch = 10
num_split = 10
weight_final = "modelActivity01.model"

#training
train=pd.read_csv('D:/user/Documents/Skripsi/Dataset/train_new.csv')

#path
image_path = 'D:/user/Documents/Skripsi/Dataset/train/'
model_path = 'D:/user/Documents/Skripsi/Model/'

# creating an empty list
train_image = []

for i in tqdm(range(train.shape[0])):
    # loading the image and keeping the target size as (224,224,3)
    img = image.load_img(os.path.join(image_path, train['image'][i]), target_size=(224,224,3))
    # converting it to array
    img = image.img_to_array(img)
    # normalizing the pixel value
    img = img/255
    # appending the image to the train_image list
    train_image.append(img)
    
# converting the list to numpy array
X = np.array(train_image)

# shape of the array
X.shape

# separating the target
y = train['class']

# creating the training and validation set
trainX, testX, trainY, testY = train_test_split(X, y, random_state=42, test_size=0.25, stratify = y)

if(num_split > 0):
    trainX = np.array_split(trainX, num_split)[0]
    trainY = np.array_split(trainY, num_split)[0]
    testX = np.array_split(testX, num_split)[0]
    testY = np.array_split(testY, num_split)[0]

#initialize the training data augmentation object
train_aug = ImageDataGenerator(
    rotation_range = 30,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.15,
    zoom_range = 0.15,
    fill_mode = 'nearest'
    )
train_batches = train_aug.flow(trainX, trainY, batchsize=32)

#initialize the validation/testing data augmentation
val_aug = ImageDataGenerator()
val_batches = val_aug.flow(testX, testY, batchsize=32)

#load ResNet-50 network
baseModel = ResNet50(
    weights='imagenet',
    include_top=False, 
    input_tensor=Input(shape=224,224,3)
    )

# construct the head of model
headModel = baseModel.output
headModel = Flatten(name='flatten')(baseModel)
headModel = Dropout(0.5)(baseModel)
headModel = Dense(np.unique(trainY).shpe[0], activation='softmax')(baseModel)

#place head model on top of base model
model = Model(inputs=baseModel.input, output=headModel)

# loop over all layers in the base model and freeze them so they wont be updated during the training process
for layer in baseModel.layers:
    layer.trainable = False

model.summary()

# compile our model (this needs to be done after our setting our layers to being non-trainable)
print("[INFO] compiling model...")
model.compile(optimizer=Adabound(lr=1e-3, final_lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])


#train the head of the network for a few epochs (all other layers are frozen) -- this will allow the new FC layers to start to become initialized with actual "learned" values versus pure random
print("[INFO] training head...")
H = model.fit(
    x=train_batches, 
    steps_per_epoch=len(trainX) // 32,
    validation_data=val_batches,
    validation_steps=len(testX) // 32,
    epochs=num_epoch
    )

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX.astype('float32'), batch_size=32)
print("confusion Metrix"),
print(confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1)))
print("classification report"),
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=y))
score = model.evaluate(val_batches, steps=1000, verbose=1)
print("Accuracy is %s " % (score[1]*100))

# plot the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, num_epoch), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, num_epoch), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, num_epoch), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, num_epoch), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('plot.png')

print("[INFO] serializing network...")
model.save(os.path.join(model_path, weight_final), save_format="h5")