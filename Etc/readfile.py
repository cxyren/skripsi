# try:
#     file = open("D:\\user\\Documents\\Skripsi\\Dataset\\k28dtm7tr6-1\\k28dtm7tr6-1\\a01_s01_e01_screen.txt")
#     print(file.read())
# except:
#     print("ERROR")

from PIL import Image
import numpy as np

data = np.loadtxt("D:\\user\\Documents\\Skripsi\\Dataset\\k28dtm7tr6-1\\k28dtm7tr6-1\\a18_s01_e01_screen.txt")
data = data.astype(np.uint64)
# data2 = np.genfromtxt("D:\\user\\Documents\\Skripsi\\Dataset\\k28dtm7tr6-1\\k28dtm7tr6-1\\a01_s01_e01_screen.txt")  
print(data)
#print(data2)
print(data.dtype)
print(data.shape)
print(data.strides)