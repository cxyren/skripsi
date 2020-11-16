import numpy as np
import os
import cv2

filename = "D:/user/Documents/Skripsi/Dataset/ntu-skeleton/skeletons/S001C001P001R001A001.skeleton"

def read_skeleton_file(filename):
    file = open(filename)
    framecount = file.readline()

    bodyinfo = []
    for i in range(int(framecount)):
        bodycount = file.readline()
        bodies = []
        for j in range(int(bodycount)):
            arraynum = file.readline().split()
            body = {
                "bodyID": arraynum[0],
                "clipedEdges": arraynum[1],
                "handLeftConfidence": arraynum[2],
                "handLeftState": arraynum[3],
                "handRightConfidence": arraynum[4],
                "handRightState": arraynum[5],
                "isResticted": arraynum[6],
                "leanX": arraynum[7],
                "leanY": arraynum[8],
                "trackingState": arraynum[9],
                "jointCount": file.readline(),
                "joints": []
            }
            for k in range(int(body["jointCount"])):
                jointinfo = file.readline().split()
                joint={
                    "x": jointinfo[0],
                    "y": jointinfo[1],
                    "z": jointinfo[2],
                    "depthX": jointinfo[3],
                    "depthY": jointinfo[4],
                    "colorX": jointinfo[5],
                    "colorY": jointinfo[6],
                    "orientationW": jointinfo[7],
                    "orientationX": jointinfo[8],
                    "orientationY": jointinfo[9],
                    "orientationZ": jointinfo[10],
                    "trackingState": jointinfo[11]
                }
                body["joints"].append(joint)
            bodies.append(body)
        bodyinfo.append(bodies)
    file.close()
    return bodyinfo

def bresenham_line(x0, y0, x1, y1):
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0  
        x1, y1 = y1, x1

    switched = False
    if x0 > x1:
        switched = True
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    if y0 < y1: 
        ystep = 1
    else:
        ystep = -1

    deltax = x1 - x0
    deltay = abs(y1 - y0)
    error = -deltax / 2
    y = y0

    line = []    
    for x in range(x0, x1 + 1):
        if steep:
            line.append((y,x))
        else:
            line.append((x,y))

        error = error + deltay
        if error > 0:
            y = y + ystep
            error = error - deltax
    if switched:
        line.reverse()
    return line

bodyinfo = read_skeleton_file(filename)

path = "D:/user/Documents/Skripsi/Github-Program/Etc/video/"
videoname = "sampleA001_rgb.mp4"

videofile = cv2.VideoCapture(os.path.join(path, videoname))
success,image = videofile.read()

connecting_joint = [2, 1, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 8, 8, 12, 12]

print(len(bodyinfo))
for i in range(len(bodyinfo)):
    if(success):
        print("frames %d" % i)
        for j in range(len(bodyinfo[i])):
            for k in range(25):
                # red for line
                rv = 255
                gv = 0
                bv = 0

                l = connecting_joint[k] - 1

                joint = bodyinfo[i][j]['joints'][k]
                dx = np.int32(round(float(joint['colorX'])))
                dy = np.int32(round(float(joint['colorY'])))
                joint2 = bodyinfo[i][j]['joints'][l]
                dx2 = np.int32(round(float(joint2['colorX'])))
                dy2 = np.int32(round(float(joint2['colorY'])))

                print("1st joint %d" %k)
                print(dx),
                print(dy)
                print("2nd joint %d" %l)
                print(dx2),
                print(dy2)

                line = bresenham_line(dx, dy, dx2, dy2)

                print(len(line))

                for m in range(len(line)):
                    dx = line[m][0]
                    dy = line[m][1]
                    image = cv2.circle(image, (dx, dy), radius=3, color=(bv, gv, rv), thickness=-1)

                #green color for points/ joints
                rv = 0
                gv = 255
                bv = 0

                joint = bodyinfo[i][j]['joints'][k]
                dx = np.int32(round(float(joint['colorX'])))
                dy = np.int32(round(float(joint['colorY'])))

                image = cv2.circle(image, (dx, dy), radius=5, color=(bv, gv, rv), thickness=-1)
        path = 'D:/user/Documents/Skripsi/Github-Program/Etc/Tes/'
        cv2.imwrite(os.path.join(path, videoname.split('_')[0] + "_frame%d.jpg" % i), image)
        success,image = videofile.read()

videofile.release()
        


