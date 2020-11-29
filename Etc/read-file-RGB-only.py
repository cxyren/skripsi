from glob import glob
from tqdm import tqdm
import pandas as pd

video = glob('D:/user/Documents/Skripsi/Dataset/ntu-skeleton/skeletons/*')
class_name = pd.read_csv('D:/user/Documents/Skripsi/Dataset/class_name_new.csv')
setup_num = dict()
setup_num['S003'] = True
setup_num['S004'] = True
setup_num['S005'] = True
setup_num['S006'] = True
setup_num['S008'] = True
setup_num['S009'] = True
skeleton_name = []
name_class = []

for i in tqdm(range(len(video))):
    name = video[i].split('/')[6]
    name = name[13:]
    if name.split('_')[0][:4] in setup_num:
        for j in range(class_name.shape[0]):
            if name.split('_')[0][-4:] == class_name['code'][j]:
                video_name.append(name)
                name_class.append(class_name['code'][j])
                break

df = pd.DataFrame()
df['video'] = video_name

df.to_csv('D:/user/Documents/Skripsi/Dataset/fix/RGB_newest3.csv',header=True, index=False)