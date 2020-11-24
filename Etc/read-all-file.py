from glob import glob
from tqdm import tqdm
import pandas as pd

video = glob("D:/user/Documents/Skripsi/Dataset/RGB-raw/nturgb+d_rgb/*")
video_name = []

for i in tqdm(range(len(video))):
    name = video[i].split('/')[6]
    video_name.append(name[13:])

df = pd.DataFrame()
df['video'] = video_name

df.to_csv('D:/user/Documents/Skripsi/Dataset/fix/dataset_all.csv',header=True, index=False)