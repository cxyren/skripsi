from tqdm import tqdm
import pandas as pd
import os
import numpy as np

#path
output_path = 'D:/user/Documents/Skripsi/Dataset/'
output_file = 'train_new_split2.csv'

train = pd.read_csv('D:/user/Documents/Skripsi/Dataset/train_new.csv')

image = []
label = []

for i in tqdm(range(train.shape[0])):
    if int(train['image'][i].split('_')[1].split('.')[0][5:]) > 14 and int(train['image'][i].split('_')[1].split('.')[0][5:]) < 45:
        image.append(train['image'][i])
        label.append(train['class'][i])

output = pd.DataFrame()
output['image'] = image
output['class'] = label

output.to_csv(os.path.join(output_path, output_file), header=True, index=False)