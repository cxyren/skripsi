from sklearn.model_selection import train_test_split
import os
import pandas as pd
from tqdm import tqdm

train_new_file = 'train_newest20.csv'
test_new_file = 'test_newest20.csv'

train_file_path = 'D:/user/Documents/Skripsi/Dataset/fix/'

train = pd.read_csv('D:/user/Documents/Skripsi/Dataset/fix/train_newest10.csv')
test = pd.read_csv('D:/user/Documents/Skripsi/Dataset/fix/test_newest10.csv')

name_class = pd.read_csv('D:/user/Documents/Skripsi/Dataset/class_name_new_new_new.csv')
class_code = dict()
for i in range(name_class.shape[0]):
    class_code[name_class['code'][i]] = name_class['name'][i]

dir_label = []

print(train.shape[0])

# subject_num = dict()
# subject_num['P001'] = True
# subject_num['P002'] = True
# subject_num['P004'] = True
# subject_num['P005'] = True
# subject_num['P008'] = True
# subject_num['P009'] = True
# subject_num['P013'] = True
# subject_num['P014'] = True
# subject_num['P015'] = True
# subject_num['P016'] = True
# subject_num['P017'] = True
# subject_num['P018'] = True
# subject_num['P019'] = True
# subject_num['P025'] = True
# subject_num['P027'] = True
# subject_num['P028'] = True
# subject_num['P031'] = True
# subject_num['P034'] = True
# subject_num['P035'] = True
# subject_num['P038'] = True

image = []
label = []

for i in tqdm(range(train.shape[0])):
    if not train['class'][i]:
        continue
    # print(train['skeleton'][i][8:12])
    # if train['skeleton'][i][4:8] == 'C001':
    #     label.append(train['class'][i])
    #     image.append(train['image'][i])
    #     dir_label.append('C:/new_train_crop/')
    if train['skeleton'][i][-4:] in class_code:
        label.append(train['class'][i])
        image.append(train['image'][i])
        dir_label.append('C:/new_train_crop/')

train = pd.DataFrame()
train['image'] = image
train['class'] = label
train['dir'] = dir_label
train.to_csv(os.path.join(train_file_path, train_new_file), header=True, index=False)

dir_label = []
image = []
label = []

for i in tqdm(range(test.shape[0])):
    if not test['class'][i]:
        continue
    # print(train['skeleton'][i][8:12])
    # if train['skeleton'][i][4:8] == 'C001':
    #     label.append(test['class'][i])
    #     image.append(test['image'][i])
    #     dir_label.append('C:/new_test_crop/')
    if test['image'][i].split('_')[0][-4:] in class_code:
        label.append(test['class'][i])
        image.append(test['image'][i])
        dir_label.append('C:/new_test_crop/')
    
# trainX1, testX1, trainY1, testY1 = train_test_split(train_image, label, random_state = 42, test_size = 0.5, stratify = label)

# trainX2, testX2, trainY2, testY2 = train_test_split(trainX1, trainY1, random_state = 42, test_size = 0.5, stratify = trainY1)

# trainX3, testX3, trainY3, testY3 = train_test_split(trainX2, trainY2, random_state = 42, test_size = 0.5, stratify = trainY2)

# trainX4, testX4, trainY4, testY4 = train_test_split(trainX3, trainY3, random_state = 42, test_size = 0.5, stratify = trainY3)

train = pd.DataFrame()
train['image'] = image
train['class'] = label
train['dir'] = dir_label
train.to_csv(os.path.join(train_file_path, test_new_file), header=True, index=False)

# train = pd.DataFrame()
# train['image'] = testX4
# train['class'] = testY4
# train.to_csv(os.path.join(train_file_path, train_file2), header=True, index=False)

# train = pd.DataFrame()
# train['image'] = testX3
# train['class'] = testY3
# train.to_csv(os.path.join(train_file_path, train_file3), header=True, index=False)

# train = pd.DataFrame()
# train['image'] = testX2
# train['class'] = testY2
# train.to_csv(os.path.join(train_file_path, train_file4), header=True, index=False)

# train = pd.DataFrame()
# train['image'] = testX1
# train['class'] = testY1
# train.to_csv(os.path.join(train_file_path, train_file5), header=True, index=False)