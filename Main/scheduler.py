import os
import sys

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

try:
    # os.system('python "../etc/crop-image.py"')
    os.system('python loadimage.py')
    print('COPY')
    os.system('python "train copy.py"')
    print('COPY 2')
    os.system('python "train copy 2.py"')
    print('VGG16')
    os.system('python train.py')
    os.system('shutdown -s -t 1')
    # print('26')
    # os.system('python "train copy.py"')
    # print('27')
    # os.system('python "train copy 2.py"')
except Exception as e:
    print(e)
    sys.exit()
