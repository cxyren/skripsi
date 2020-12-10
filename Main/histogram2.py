import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import os
import pickle
import math
import statistics

name_class = pd.read_csv('D:/user/Documents/Skripsi/Dataset/class_name_new.csv')
class_code = dict()
for i in range(name_class.shape[0]):
    class_code[name_class['code'][i]] = name_class['name'][i]

size = 4.0

if not os.path.isfile("C:/users/cxyre/Desktop/block1.pickle") and not os.path.isfile("C:/users/cxyre/Desktop/label1.pickle"):
    #csv
    skeleton_csv = pd.read_csv('D:/user/Documents/Skripsi/Dataset/fix/train_newest11.csv')
    
    blocks = []

    count = -1

    blocks_class = []

    count_class = dict()
    for i in range(name_class.shape[0]):
        count_class[name_class['code'][i]] = 0

    print(skeleton_csv.shape[0])
    frame_count = 0
    for i in tqdm(range(skeleton_csv.shape[0])):
        if skeleton_csv['image'][i].split('_')[0][-4:] not in class_code:
            continue

        if i % 10 == 0 :
            if count_class.get(skeleton_csv['image'][i].split('_')[0][-4:]) > 100:
                continue
            block = []
            frame_count = 0
            loop_ = int(224/size)
            for j in range(loop_):
                block.append([])
                for k in range(loop_):
                    block[j].append(0)
            blocks.append(block)
            count = count + 1
            blocks_class.append(class_code.get(skeleton_csv['image'][i].split('_')[0][-4:]))
            count_class[skeleton_csv['image'][i].split('_')[0][-4:]] += 1

        img = cv2.imread(os.path.join('C:/new_train_crop/', skeleton_csv['image'][i]))
        #making empty frame    
        frame = np.zeros(shape=[224, 224, 3], dtype=np.uint8)
        if img.shape[1] < img.shape[0]:
            img = cv2.resize(img, (int(img.shape[1]*(224/img.shape[0])), 224))

            #relocating image
            center_x = frame.shape[1] / 2
            center_x2 = img.shape[1] / 2
            
            frame[int(0):int(224), int(center_x - center_x2):int(center_x + center_x2)] = img
        else:
            img = cv2.resize(img, (224, int(img.shape[0]*(224/img.shape[1]))))

            #relocating image
            center_y = frame.shape[0] / 2
            center_y2 = img.shape[0] / 2
            
            frame[int(center_y - center_y2):int(center_y + center_y2), int(0):int(224)] = img
            
        #resize image
        frame = cv2.resize(frame, (224, 224)) 
        frame = frame.astype('float64')
        frame *= 255.0/frame.max()

        frame_count = frame_count + 1
        for j in range(frame.shape[1]):
            for k in range(frame.shape[0]):
                if blocks[count][math.ceil(k*1.0/size) - 1][math.ceil(j*1.0/size) - 1] == frame_count:
                    continue
                if frame[k, j][1] >= 150:
                    # print(frame[k, j])
                    blocks[count][math.ceil(k*1.0/size) - 1][math.ceil(j*1.0/size) - 1] = blocks[count][math.ceil(k*1.0/size) - 1][math.ceil(j*1.0/size) - 1] + 1

    f = open(os.path.join('C:/users/cxyre/Desktop/', 'block1.pickle'), "wb")
    f.write(pickle.dumps(blocks))
    f.close()

    f = open(os.path.join('C:/users/cxyre/Desktop/', 'label1.pickle'), "wb")
    f.write(pickle.dumps(blocks_class))
    f.close()

block = pickle.loads(open(os.path.join('C:/users/cxyre/Desktop/', 'block1.pickle'), "rb").read())
label = pickle.loads(open(os.path.join('C:/users/cxyre/Desktop/', 'label1.pickle'), "rb").read())

code_list = []
for key,value in class_code.items() :
    code_list.append(value)

similiar = []
for i in tqdm(range(len(code_list))):
    similiar.append([])
    for j in tqdm(range(len(label))):
        if label[j] == code_list[i]:
            for k in range(j, len(block)):
                if label[k] != code_list[i]:
                    continue
                if j == k:
                    continue
                sim = 0
                loop_ = int(224/size)
                for l in range(loop_):
                    for m in range(loop_):
                        if block[j][l][m] <= block[k][l][m] + 1 and block[j][l][m] >= block[k][l][m] - 1:
                            sim += 1
                similiar[i].append(sim * 100 / (loop_ * loop_))

average_inclass = []
min_inclass = []
max_inclass = []
median_inclass = []
for i in tqdm(range(len(similiar))):
    average_inclass.append(statistics.mean(similiar[i]))
    min_inclass.append(min(similiar[i]))
    max_inclass.append(max(similiar[i]))
    median_inclass.append(statistics.median(similiar[i]))

for i in range(len(average_inclass)):
    print('Average %i: %f%%\n' %(i, average_inclass[i]))
    print('Min %i: %f%%\n' %(i, min_inclass[i]))
    print('Min %i: %f%%\n' %(i, max_inclass[i]))
    print('Median %i: %f%%\n' %(i, median_inclass[i]))

similiar = []
for i in tqdm(range(len(code_list))):
    similiar.append([])
    for j in tqdm(range(len(block))):
        if label[j] == code_list[i]:
            for k in range(j, len(block)):
                if j == k:
                    continue
                if label[j] == label[k]:
                    continue
                sim = 0
                loop_ = int(224/size)
                for l in range(loop_):
                    for m in range(loop_):
                        if block[j][l][m] <= block[k][l][m] + 1 and block[j][l][m] >= block[k][l][m] - 1:
                            sim += 1
                similiar[i].append(sim * 100 / (loop_ * loop_))

# print('LEN: %i' % len(similiar))

average_outclass = []
for i in tqdm(range(len(similiar))):
    average_outclass.append(statistics.mean(similiar[i]))

for i in range(len(average_outclass)):
    print('Average %i: %f%%\n' %(i, average_outclass[i]))

average_total = statistics.mean(average_outclass)
print('Average between class: %f%%\n' % (average_total))