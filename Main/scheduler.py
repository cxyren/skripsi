import os
import sys

try:
    print('19')
    os.system('python train.py')
    print('20')
    os.system('python train.py')
    print('21')
    os.system('python train-copy1.py')
    print('22')
    os.system('python train-copy2.py')
    print('23')
    os.system('python train-copy3.py')
    print('24')
    os.system('python train-copy.py')
    print('25')    
    os.system('python train.py')
except Exception as e:
    print(e)
    sys.exit()