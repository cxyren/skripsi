import os
import sys

try:
    os.system('python train.py')
    os.system('python train.py')
except Exception as e:
    print(e)
    sys.exit()