from glob import glob
from tqdm import tqdm
import pandas as pd

video = glob("D:/user/Documents/Skripsi/Dataset/train/*")
class_name = pd.read_csv('D:/user/Documents/Skripsi/Dataset/class_name.csv')
image_name = []
name_class = []

for i in tqdm(range(len(video))):
    name = video[i].split('/')[5]
    name = name[6:]
    for j in range(class_name.shape[0]):
        if(name.split('_')[0][-4:] == class_name['code'][j]):
            image_name.append(name)
            name_class.append(class_name['code'][j])
            break

df = pd.DataFrame()
df['image'] = image_name
df['class'] = name_class

df.to_csv('D:/user/Documents/Skripsi/Dataset/fix/train_new.csv',header=True, index=False)