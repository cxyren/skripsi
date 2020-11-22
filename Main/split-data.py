from sklearn.model_selection import train_test_split
import os
import pandas as pd
from tqdm import tqdm

train_file1 = 'train_1.csv'
train_file2 = 'train_2.csv'

train_file_path = 'D:/user/Documents/Skripsi/Dataset/'

train = pd.read_csv('D:/user/Documents/Skripsi/Dataset/train_new.csv')

print(train.shape[0])

train_image = []
label = []

for i in tqdm(range(train.shape[0])):
    if not train['class'][i]:
        continue
    
    train_image.append(train['image'][i])
    if train['image'][i].split('_')[0][-4:] == 'A055':
        label.append('hugging other person.')
    elif train['image'][i].split('_')[0][-4:] == 'A056':
        label.append('giving something to other person.')
    else:
        label.append(train['class'][i])

trainX, testX, trainY, testY = train_test_split(train_image, label, random_state = 42, test_size = 0.5, stratify = label)

train1 = pd.DataFrame()
train1['image'] = trainX
train1['class'] = trainY
train1.to_csv(os.path.join(train_file_path, train_file1), header=True, index=False)

train2 = pd.DataFrame()
train2['image'] = testX
train2['class'] = testY
train2.to_csv(os.path.join(train_file_path, train_file2), header=True, index=False)