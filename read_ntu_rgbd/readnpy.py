import numpy as np

skeleton = np.load('D:\\user\\Documents\\Skripsi\\Program\\read_ntu_rgbd\\raw_npy\\S001C001P001R001A001.skeleton.npy', allow_pickle = True) 

print(skeleton)
print(np.info(skeleton))