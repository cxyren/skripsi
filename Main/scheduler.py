import os
import sys

try:
    os.system('python read-file-RGB-only.py')
    os.system('python pre-processing.py')
    os.system('python train.py')
except Exception as e:
    print(e)
    sys.exit()