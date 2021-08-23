import os, glob
import cv2
import numpy as np
# import matplotlib.pyplot as plt

def mkfolder(folder):
    if not os.path.lexists(folder):
        os.makedirs(folder)
    
def imshow_plt(img):
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap ='gray')
    plt.show()
    
def randomRoate(img, label, p, angle_range):
    if len(img.shape)==3:
        height, width, channel = img.shape
    else:
        height, width = img.shape
    
    random_angle = np.random.randint(-angle_range/2, angle_range/2)*2
#     print(random_angle)
    matrix = cv2.getRotationMatrix2D((width/2, height/2), random_angle, 1)
    rotate_img = cv2.warpAffine(img, matrix, (width, height))
    rotate_label = cv2.warpAffine(label, matrix, (width, height))
#     imshow_plt(rotate_img)
#     imshow_plt(rotate_label)
    
    return rotate_img, rotate_label

def find_top_point(ori_img, mask_img):
    mask_img_T = mask_img.T

    tmp_index=[]
    for i, c in enumerate(mask_img_T):
        if len(np.unique(c))>1:
            tmp_index.append(i)
    x_min = min(tmp_index)
    x_max = max(tmp_index)
#     print(x_min, x_max)

    p_x = int((x_min+x_max)/2) 
    p_y = np.where(mask_img_T[int((x_min+x_max)/2)]==255)[0][0]
    p = (p_x, p_y)
#     cv2.line(ori_img, p, p, (255,0,0), 5)
#     imshow_plt(ori_img)
    
    return p_x, p_y

def Augment_crop(img, mask):
    p_x, p_y=find_top_point(ori_img, mask_img)
    rotate_img, rotate_mask = randomRoate(img, mask, (p_x, p_y), 20)
    random_size = np.random.randint(15,35)*20 # 300-600
#     print(random_size)
    
    h, w= img.shape
    x1 = p_x-random_size if p_x-random_size>0 else 0
    x2 = p_x+random_size if p_x+random_size<w else w
    y1 = p_y-random_size if p_y-random_size>0 else 0
    y2 = p_y+random_size if p_y+random_size<h else h
#     print(h,w)
#     print(x1,x2,y1,y2)
    crop_img = rotate_img[y1:y2,x1:x2]
    crop_mask = rotate_mask[y1:y2,x1:x2]
#     imshow_plt(crop_img)
#     imshow_plt(crop_mask)
#     print(crop_mask.shape)
#     if crop_img==[] or crop_mask==[]:
#         imshow_plt(ori_img)
        
    return crop_img, crop_mask

img_path = '../3_deepdata/1_exp1/train/lat_pre_img/'
mask_path = '../3_deepdata/1_exp1/train/lat_pre_label/'

img_li = sorted(glob.glob(img_path+'*.png'))
mask_li = sorted(glob.glob(mask_path+'*png'))

print(len(img_li), len(mask_li))

i=0

mkfolder('../3_deepdata/1_exp1/train/LAT_aug_img/')
mkfolder('../3_deepdata/1_exp1/train/LAT_aug_label/')
for img, mask in zip(img_li, mask_li):
#     if i==10:
#         break
    if i%100==0:
        print('{}/{}'.format(i, len(img_li)))
        
    ori_img = cv2.imread(img, 0)
    mask_img = cv2.imread(mask, 0)
    img_name = img[img.rindex('/')+1:-4]
#     print(img_name)
    
    cv2.imwrite('../3_deepdata/1_exp1/train/LAT_aug_img/{}.png'.format(img_name), cv2.resize(ori_img, (512,512)))
    cv2.imwrite('../3_deepdata/1_exp1/train/LAT_aug_label/{}.png'.format(img_name), cv2.resize(mask_img, (512,512)))
    for j in range(9):
        aug_img, aug_mask = Augment_crop(ori_img, mask_img)        
        aug_img = cv2.resize(aug_img, (512,512))
        aug_mask = cv2.resize(aug_mask, (512,512))
        cv2.imwrite('../3_deepdata/1_exp1/train/LAT_aug_img/{}_{}.png'.format(img_name, j), aug_img)
        cv2.imwrite('../3_deepdata/1_exp1/train/LAT_aug_label/{}_{}.png'.format(img_name, j), aug_mask)
#     imshow_plt(ori_img)
    
    i+=1