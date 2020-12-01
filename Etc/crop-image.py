import pandas as pd
from tqdm import tqdm
from glob import glob
import os
import cv2

#training data
df = pd.read_csv('D:/user/Documents/Skripsi/Dataset/fix/train_newest8.csv')
name_class = pd.read_csv('D:/user/Documents/Skripsi/Dataset/class_name_new.csv')

dest_path = 'C:/train_tests_crop/'
arr_path = 'C:/train_tests/'

train_image = []
train_class = []
print("[INFO] load image ...")
for i in tqdm(range(df.shape[0])):
    if not df['class'][i]:
        continue
    for j in range(name_class.shape[0]):
        if df['class'][i] == name_class['name'][j]:
            # loading the image and resize to 224x224 and rgb
            img = cv2.imread(os.path.join(arr_path, df['image'][i]))
            
            left = 9999
            right = -1
            bot = 0

            for j in range(img.shape[1]):
                for k in range(img.shape[0]):
                    if all(l > 0 for l in img[k,j]):
                        if k > bot:
                            bot = k
                        if j < left:
                            left = j
                        if j > right:
                            right = j

            right = right + 2
            left = left - 2
            bot = bot + 2
            top = bot - 165

            crop_img = img[int(top):int(bot), int(left):int(right)]
            # save img
            cv2.imwrite(os.path.join(dest_path, df['image'][i]), crop_img)
            
# del df

# print('[INFO]PLACING LABEL INTO IMAGE...')
# # getting the names of all the images
# images = glob(os.path.join(dest_path, '*'))
# name_class = pd.read_csv('D:/user/Documents/Skripsi/Dataset/class_name_new.csv')
# train_image = []
# train_class = []

# for i in tqdm(range(len(images))):
#     # creating the image name
#     _nameimage = images[i].split('/')[1]
#     # creating the class of image 
#     _class = _nameimage.split('_')[2][-4:]
#     for j in range(name_class.shape[0]):
#         if _class == name_class['code'][j]:            
#             train_image.append(_nameimage[17:])
#             train_class.append(name_class['name'][j])
#             break
    
# # storing the images and their class in a dataframe
# train_data = pd.DataFrame()
# train_data['image'] = train_image
# train_data['class'] = train_class

# print('[INFO]SAVING INTO CSV...')
# # converting the dataframe into csv file 
# train_data.to_csv('D:/user/Documents/Skripsi/Dataset/fix/train_newest7.csv', header=True, index=False)