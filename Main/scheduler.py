import os
import sys
import time
from tqdm import tqdm
# try:
#     os.system('python loadimage.py')
#     print('19')
#     os.system('python train.py')
#     print('20')
#     os.system('python train-copy.py')
#     print('21')
#     os.system('python train-copy1.py')
#     print('22')
#     os.system('python train-copy2.py')
#     print('23')
#     os.system('python train-copy3.py')
#     print('24')
#     os.system('python train.py')
#     print('25')    
#     os.system('python train-copy.py')
# except Exception as e:
#     print(e)
#     sys.exit()

# try:
#     # os.system('python "../etc/crop-image.py"')
#     os.system('python loadimage.py')
#     print('COPY')
#     os.system('python "train copy.py"')
#     print('COPY 2')
#     os.system('python "train copy 2.py"')
#     print('VGG16')
#     os.system('python train.py')
#     for i in tqdm(range(300)):
#         time.sleep(1)
#     os.system('shutdown -s -t 300')
#     # print('26')
#     # os.system('python "train copy.py"')
#     # print('27')
#     # os.system('python "train copy 2.py"')
# except Exception as e:
#     print(e)
#     sys.exit()


try:
    subject = {
        '0':3,
        '1':4,
        '2':5,
        '3':6,
        '4':8
    }
#    activity = {
#     #    '0':1,
#        '1':2,
#        '2':8,
#     #    '3':9,
#     #    '4':11,
#     #    '5':12,
#        '6':37,
#     #    '7':41,
#        '8':43,
#     #    '9':44,
#        '10':45
#     #    '11':46,
#     #    '12':47,
#    }
    activity = {
        '0':2,
        '1':8,
        '2':37,
        '3':43,
        '4':45
    }
    for i in range(5):
        for j in range(5):
                os.system('python predict-video.py csv/S0%iA%02i.csv'%(subject.get(str(i)), activity.get(str(j))))
except Exception as e:
    print(e)
    sys.exit()

# try:
#     # for i in tqdm(range(900)):
#     #     time.sleep(1)
#     os.system('python histogram.py')
#     for i in tqdm(range(300)):
#         time.sleep(1)
#     os.system('shutdown -s -t 1')
# except Exception as e:
#     print(e)
#     f = open(os.path.join('C:/users/cxyre/desktop/', 'error.txt'), 'w')
#     f.write(e)
#     f.close()
#     os.system('shutdown -s -t 1')
#     sys.exit()

# try:
    
#     # os.system('python pre-processing.py')
#     # os.system('python "../etc/crop-image.py"')
#     # os.system('python "../etc/crop-image.py"')
#     # os.system('python loadimage.py')
#     # os.system('python loadimage.py')
#     # os.system('python histogram.py')
#     # print('COPY')
#     os.system('python "train copy 3.py"')
#     # print('COPY 2')
#     # os.system('python "train copy 2.py"')
#     # print('VGG16')
#     # os.system('python train.py')
#     # for i in tqdm(range(300)):
#     #     time.sleep(1)
#     os.system('shutdown -s -t 300')
#     # print('26')
#     # os.system('python "train copy.py"')
#     # print('27')
#     # os.system('python "train copy 2.py"')
# except Exception as e:
#     print(e)
#     sys.exit()