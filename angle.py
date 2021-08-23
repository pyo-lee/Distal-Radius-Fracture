import cv2
import numpy as np
import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import math

def max_contour(img):
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray,127,255,0)
    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #print(len(contours))

    contour = []
    tmp =[]

    for k in range(len(contours)):
        cnt = contours[k]
        mmt = cv2.moments(cnt) 
        """for key, value in mmt.items(): 
            print(key," : ",value)"""
        tmp.append(float(mmt['m00']))

    max_num = max(tmp)
    index = tmp.index(max_num)
    #print(index)
    contour.append(contours[index])

#     image = cv2.drawContours(img, contour, -1, (0,255,0), 2)
#     image = array_to_img(image)
#     image.save("./check/1_contour.png")
    
    return contour

def find_ymax_point(contour):
    x_list, y_list, xy_list =[], [], ()

    for n in range(len(contour)):
        x_list, y_list, xy_list =[], [], ()
        num_contour = contour[n]
        list_contour=num_contour.tolist()

        for i in range(len(list_contour)):
            x, y = list_contour[i][0]
            y_list.append(y)
            x_list.append(x)

        index = y_list.index(min(y_list))
        point_1 = tuple(list_contour[index][0])
        
    return point_1

def find_center_point(contour):
    c_point = []
    a=contour[0].tolist()
    x_list, y_list =[], []

    for i in range(len(a)):
        x, y = a[i][0]
        y_list.append(y)
        x_list.append(x)

    xy_list=[]

    for i in range(len(x_list)):
        xy = x_list[i], y_list[i]
        #xy.tolist()
        xy_list.append(xy)

    same_y = []

    n_y = set(y_list)

    for i in n_y:
        tmp=[]
        for k in range(1,len(y_list)):
            if i == y_list[k]:
                #print(xy_list[k])
                if not i in tmp:
                    tmp.append(xy_list[k])            
        same_y.append(tmp)
    tmp=[]
    for k in range(len(same_y)):
        try:
            mid_x = same_y[k][0][0]+same_y[k][-1][0]
            mid = (int(mid_x/2), same_y[k][0][1])
            #print(mid)
            c_point.append(mid)

        except:
            pass
        
    center_x = c_point[int(len(c_point)*0.3)]
    center_x2 = c_point[int(len(c_point)*0.3)+60]
    return x_list, y_list, c_point, center_x, center_x2

def find_rotate_center_point(contour):
    c_point = []
    a=contour[0].tolist()
    x_list, y_list =[], []

    for i in range(len(a)):
        x, y = a[i][0]
        y_list.append(y)
        x_list.append(x)

    xy_list=[]

    for i in range(len(x_list)):
        xy = x_list[i], y_list[i]
        #xy.tolist()
        xy_list.append(xy)

    same_y = []

    n_y = set(y_list)

    for i in n_y:
        tmp=[]
        for k in range(1,len(y_list)):
            if i == y_list[k]:
                #print(xy_list[k])
                if not i in tmp:
                    tmp.append(xy_list[k])            
        same_y.append(tmp)
    tmp=[]
    for k in range(len(same_y)):
        try:
            mid_x = same_y[k][0][0]+same_y[k][-1][0]
            mid = (int(mid_x/2), same_y[k][0][1])
            #print(mid)
            c_point.append(mid)

        except:
            pass

    
    center_x = c_point[int(len(c_point)*0.3)]
    center_x2 = c_point[int(len(c_point)*0.3)+60]
            
    return x_list, y_list, center_x, center_x2

def rotate(img, angle):
    if len(img.shape)==3:
        height, width, channel = img.shape
    else:
        height, width = img.shape
    matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    rot_img = cv2.warpAffine(img, matrix, (width, height))
    return rot_img

def find_xpoint2_v2(img, contour, point_1, LR):
    
    if len(contour)==1:
        cnt=contour[0]
    x, y, w, h = cv2.boundingRect(cnt)
    
    if LR==0:
        std_point = (x+w, y)
    elif LR==1:
        std_point = (x, y)
#     print(std_point)
#     cv2.line(img, std_point, std_point, (0,255,255), 2)
#         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
    for i, p in enumerate(cnt):
        if LR==0:
            distance = math.sqrt(((p[0][0]-std_point[0])**2)+((p[0][1]-std_point[1])**2))
        else:
            distance = math.sqrt(((p[0][0]-std_point[0])**2)+((p[0][1]-std_point[1])**2))
        if i==0:
            min_dist = (p, distance)
        elif (i>0 and distance<min_dist[1]):
            p=p[0]
            min_dist = [p, distance]

    p1 = tuple(min_dist[0])
    
    if abs(point_1[0]-p1[0])<30:
        if LR==0:
            std_point = (x, y)
        elif LR==1:
            std_point = (x+w, y)
        
        for i, p in enumerate(cnt):
            if LR==0:
                distance = math.sqrt(((p[0][0]-std_point[0])**2)+((p[0][1]-std_point[1])**2))
            if i==0:
                min_dist = (p, distance)
            elif (i>0 and distance<min_dist[1]):
                p2 =p[0]
                min_dist = [p2, distance]
                
    p2 = tuple(min_dist[0])
    
    if abs(point_1[0]-p2[0])<20:
        p = p1
    else:
        p = p2
        
#     cv2.line(img, p, p, (255,0,0), 5)
#     cv2_imshow(img)
    
    return p, std_point
        
def left_right(point, c_point):
    #print(point[0], c_point[0])
    if point[0] > c_point[0]:
        return 1 # right
    else:
        return 0
    
def angle_between_vectors_degrees(u, v):
    """Return the angle between two vectors in any dimension space,
    in degrees."""
#     print(math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))
    return np.degrees(math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang



# def left_right2(img):