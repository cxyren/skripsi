from glob import glob
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

video = glob("D:/user/Documents/Skripsi/Dataset/RGB-raw/nturgb+d_rgb/*")
class_name = pd.read_csv('D:/user/Documents/Skripsi/Dataset/class_name.csv')
video_name = []
name_class = []

for i in tqdm(range(len(video))):
    name = video[i].split('/')[6]
    name = name[13:]
    for j in range(class_name.shape[0]):
        if(name.split('_')[0][-4:] == class_name['code'][j]):
            video_name.append(name)
            name_class.append(class_name['code'][j])
            break

trainX, testX, trainY, testY = train_test_split(video_name, name_class, random_state=42, test_size=0.3, stratify = name_class)

df = pd.DataFrame()
df['video'] = trainX

df.to_csv('D:/user/Documents/Skripsi/Dataset/fix/RGB_new.csv',header=True, index=False)