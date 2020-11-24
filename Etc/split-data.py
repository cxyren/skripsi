from sklearn.model_selection import train_test_split
import os
import pandas as pd
from tqdm import tqdm

train_file1 = 'train_1.csv'
train_file2 = 'train_2.csv'
train_file3 = 'train_3.csv'
train_file4 = 'train_4.csv'
train_file5 = 'train_5.csv'

train_file_path = 'D:/user/Documents/Skripsi/Dataset/csv/'

train = pd.read_csv('D:/user/Documents/Skripsi/Dataset/train_new.csv')

print(train.shape[0])

train_image = []
label = []

for i in tqdm(range(train.shape[0])):
    if not train['class'][i]:
        continue
    
    if train['image'][i].split('_')[0][-4:] == 'A013':
        continue
    elif train['image'][i].split('_')[0][-4:] == 'A025':
        continue
    elif train['image'][i].split('_')[0][-4:] == 'A027':
        continue
    elif train['image'][i].split('_')[0][-4:] == 'A054':
        continue
    elif train['image'][i].split('_')[0][-4:] == 'A055':
        continue
    elif train['image'][i].split('_')[0][-4:] == 'A056':
        continue
    else:
        label.append(train['class'][i])
    train_image.append(train['image'][i])

trainX1, testX1, trainY1, testY1 = train_test_split(train_image, label, random_state = 42, test_size = 0.5, stratify = label)

trainX2, testX2, trainY2, testY2 = train_test_split(trainX1, trainY1, random_state = 42, test_size = 0.5, stratify = trainY1)

trainX3, testX3, trainY3, testY3 = train_test_split(trainX2, trainY2, random_state = 42, test_size = 0.5, stratify = trainY2)

trainX4, testX4, trainY4, testY4 = train_test_split(trainX3, trainY3, random_state = 42, test_size = 0.5, stratify = trainY3)

train = pd.DataFrame()
train['image'] = trainX4
train['class'] = trainY4
train.to_csv(os.path.join(train_file_path, train_file1), header=True, index=False)

train = pd.DataFrame()
train['image'] = testX4
train['class'] = testY4
train.to_csv(os.path.join(train_file_path, train_file2), header=True, index=False)

train = pd.DataFrame()
train['image'] = testX3
train['class'] = testY3
train.to_csv(os.path.join(train_file_path, train_file3), header=True, index=False)

train = pd.DataFrame()
train['image'] = testX2
train['class'] = testY2
train.to_csv(os.path.join(train_file_path, train_file4), header=True, index=False)

train = pd.DataFrame()
train['image'] = testX1
train['class'] = testY1
train.to_csv(os.path.join(train_file_path, train_file5), header=True, index=False)