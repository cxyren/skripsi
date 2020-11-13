from PIL import Image
import numpy as np

#load file
data = np.loadtxt("D:\\user\\Documents\\Skripsi\\Dataset\\k28dtm7tr6-1\\k28dtm7tr6-1\\a18_s01_e01_screen.txt")
#change type
data = data.astype(np.uint64)
print(data)
print(data.dtype)
print(data.shape)
print(data.strides)

#width heigh
w, h = 640, 480
#series frame
j = 1
for i in range(0,data.shape[0],15):
    #clear array
    data1 = np.zeros((h, w, 3), dtype=np.uint8)
    #head neck
    if(data[i][0]>data[i+1][0]): 
        if(data[i][1]<data[i+1][1]):
            data1[data[i][1]:data[i+1][1], data[i+1][0]:data[i][0]] = [0, 255, 0]
        else:
            data1[data[i+1][1]:data[i][1], data[i+1][0]:data[i][0]] = [0, 255, 0]
    elif(data[i][0]<data[i+1][0]):
        if(data[i][1]<data[i+1][1]):
            data1[data[i][1]:data[i+1][1], data[i+1][0]:data[i][0]] = [0, 255, 0]
        else:
            data1[data[i+1][1]:data[i][1], data[i+1][0]:data[i][0]] = [0, 255, 0]
    else:
        if(data[i][1]<data[i+1][1]):
            data1[data[i][1]:data[i+1][1], data[i][0]] = [0, 255, 0]
        else:
            data1[data[i+1][1]:data[i][1], data[i][0]] = [0, 255, 0]
    #shoulders
    if(data[i+2][1]<data[i+5][1]):
        data1[data[i+2][1]:data[i+5][1], data[i+2][0]:data[i+5][0]] = [0, 255, 0]
    elif(data[i+2][1]>data[i+5][1]):
        data1[data[i+5][1]:data[i+2][1], data[i+2][0]:data[i+5][0]] = [0, 255, 0] 
    else:      
        data1[data[i+2][1], data[i+2][0]:data[i+5][0]] = [0, 255, 0] 
    #right arm
    if(data[i+2][0]>data[i+3][0]):
        if(data[i+2][1]<data[i+3][1]):
            data1[data[i+2][1]:data[i+3][1], data[i+3][0]:data[i+2][0]] = [0, 255, 0] 
        else:
            data1[data[i+3][1]:data[i+2][1], data[i+3][0]:data[i+2][0]] = [0, 255, 0] 
    else:
        if(data[i+2][1]<data[i+3][1]):
            data1[data[i+2][1]:data[i+3][1], data[i+2][0]:data[i+3][0]] = [0, 255, 0] 
        else:
            data1[data[i+3][1]:data[i+2][1], data[i+2][0]:data[i+3][0]] = [0, 255, 0]
    #right hand
    if(data[i+3][0]>data[i+4][0]):
        if(data[i+3][1]<data[i+4][1]):
            data1[data[i+3][1]:data[i+4][1], data[i+4][0]:data[i+3][0]] = [0, 255, 0] 
        else:
            data1[data[i+4][1]:data[i+3][1], data[i+4][0]:data[i+3][0]] = [0, 255, 0] 
    else:
        if(data[i+3][1]<data[i+4][1]):
            data1[data[i+3][1]:data[i+4][1], data[i+3][0]:data[i+4][0]] = [0, 255, 0] 
        else:
            data1[data[i+4][1]:data[i+3][1], data[i+3][0]:data[i+4][0]] = [0, 255, 0]
    #left arm
    if(data[i+5][0]>data[i+6][0]):
        if(data[i+5][1]<data[i+6][1]):
            data1[data[i+5][1]:data[i+6][1], data[i+6][0]:data[i+5][0]] = [0, 255, 0] 
        else:
            data1[data[i+6][1]:data[i+5][1], data[i+6][0]:data[i+5][0]] = [0, 255, 0] 
    else:
        if(data[i+5][1]<data[i+6][1]):
            data1[data[i+5][1]:data[i+6][1], data[i+5][0]:data[i+6][0]] = [0, 255, 0] 
        else:
            data1[data[i+6][1]:data[i+5][1], data[i+5][0]:data[i+6][0]] = [0, 255, 0]
    #left hand
    if(data[i+6][0]>data[i+7][0]):
        if(data[i+6][1]<data[i+7][1]):
            data1[data[i+6][1]:data[i+7][1], data[i+7][0]:data[i+6][0]] = [0, 255, 0]
        else:
            data1[data[i+7][1]:data[i+6][1], data[i+7][0]:data[i+6][0]] = [0, 255, 0]
    else:
        if(data[i+6][1]<data[i+7][1]):
            data1[data[i+6][1]:data[i+7][1], data[i+6][0]:data[i+7][0]] = [0, 255, 0] 
        else:
            data1[data[i+7][1]:data[i+6][1], data[i+6][0]:data[i+7][0]] = [0, 255, 0]
    #body
    if(data[i+1][0]>data[i+8][0]):
        data1[data[i+1][1]:data[i+8][1], data[i+8][0]:data[i+1][0]] = [0, 255, 0]
    elif(data[i+1][0]<data[i+8][0]):
         data1[data[i+1][1]:data[i+8][1], data[i+1][0]:data[i+8][0]] = [0, 255, 0]
    else:
        data1[data[i+1][1]:data[i+8][1], data[i+1][0]] = [0, 255, 0]
    #torso
    if(data[i+1][0]>data[i+8][0]):
        data1[data[i+8][1]:data[i+9][1], data[i+8][0]:data[i+1][0]] = [0, 255, 0]
    elif(data[i+1][0]<data[i+8][0]):
        data1[data[i+8][1]:data[i+9][1], data[i+1][0]:data[i+8][0]] = [0, 255, 0]
    else:
        data1[data[i+8][1]:data[i+9][1], data[i+1][0]] = [0, 255, 0]
    #hips
    if(data[i+9][1]<data[i+12][1]):
        data1[data[i+9][1]:data[i+12][1], data[i+9][0]:data[i+12][0]] = [0, 255, 0]
    elif(data[i+9][1]>data[i+12][1]):
        data1[data[i+12][1]:data[i+9][1], data[i+9][0]:data[i+12][0]] = [0, 255, 0] 
    else:
        data1[data[i+9][1], data[i+9][0]:data[i+12][0]] = [0, 255, 0] 
    #right thight
    if(data[i+9][0]>data[i+10][0]): 
        data1[data[i+9][1]:data[i+10][1], data[i+10][0]:data[i+9][0]] = [0, 255, 0]
    elif(data[i+9][0]<data[i+10][0]):
        data1[data[i+9][1]:data[i+10][1], data[i+9][0]:data[i+10][0]] = [0, 255, 0]
    else:
        data1[data[i+9][1]:data[i+10][1], data[i+9][0]] = [0, 255, 0]
    #right leg
    if(data[i+10][0]>data[i+11][0]):
        data1[data[i+10][1]:data[i+11][1], data[i+11][0]:data[i+10][0]] = [0, 255, 0]
    elif(data[i+10][0]<data[i+11][0]):
        data1[data[i+10][1]:data[i+11][1], data[i+10][0]:data[i+11][0]] = [0, 255, 0]
    else:  
        data1[data[i+10][1]:data[i+11][1], data[i+11][0]] = [0, 255, 0]
    #left thight
    if(data[i+12][0]>data[i+13][0]): 
        data1[data[i+12][1]:data[i+13][1], data[i+13][0]:data[i+12][0]] = [0, 255, 0] 
    elif(data[i+12][0]<data[i+13][0]):
        data1[data[i+12][1]:data[i+13][1], data[i+12][0]:data[i+13][0]] = [0, 255, 0]
    else:
        data1[data[i+12][1]:data[i+13][1], data[i+12][0]] = [0, 255, 0] 
    #left leg
    if(data[i+13][0]>data[i+14][0]):
        data1[data[i+13][1]:data[i+14][1], data[i+14][0]:data[i+13][0]] = [0, 255, 0]
    elif(data[i+13][0]<data[i+14][0]):
        data1[data[i+13][1]:data[i+14][1], data[i+13][0]:data[i+14][0]] = [0, 255, 0]
    else:
        data1[data[i+13][1]:data[i+14][1], data[i+13][0]] = [0, 255, 0]
    #clear image
    img = Image.new("RGB", (640, 480))
    #get image from array
    img = Image.fromarray(data1, 'RGB')
    #save image
    img.save('Image from Skeleton data\\A18S01\\frame'+str(j)+'.png')
    #increment for name
    j = j + 1
j = 0
for i in range(30, 60):
    print(str(j)+" x:"+str(data[i][0])+" y:"+str(data[i][1]))
    if(j==14):
        print("change frame")
        j = 0
    else:
        j = j + 1