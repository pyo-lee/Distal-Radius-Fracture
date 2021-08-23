import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import glob
import shutil
import cv2

def data_spilt(ori_path, mask_path, data_path, label_path, test_path, test_label_path, img_type):
    imgs = sorted(glob.glob(ori_path + "/*." + img_type))
    masks = sorted(glob.glob(mask_path + "/*." + img_type))
    index = int(len(imgs)*0.8)
    
    print('train : test = 8:2')
    print('train data : {}'.format(index))
    print('test data : {}'.format(int(len(imgs)-index)))
    
    print('-'*30)
    print('train_split...')
    print('-'*30)
    
    for img, mask in zip(imgs[:index], masks[:index]):
        shutil.copy(img, data_path)
        shutil.copy(mask, label_path)
    
    print('-'*30)
    print('train_split done...')
    print('-'*30)
    print('-'*30)
    print('test_split...')
    print('-'*30)

    for img, mask in zip(imgs[index:], masks[index:]):
        shutil.copy(img, test_path)
        shutil.copy(mask, test_label_path)
    print('-'*30)
    print('test_split done...')
    print('-'*30)
    
def create_train_data(data_path, label_path, npy_path, out_rows, out_cols, name, img_type):
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    
    imgs = glob.glob(data_path + "/*." + img_type)
    imgdatas = np.ndarray((len(imgs),out_rows,out_cols,1), dtype=np.uint8)
    imglabels = np.ndarray((len(imgs),out_rows,out_cols,1), dtype=np.uint8)

    for i, imgname in enumerate(imgs):
        if i%1000==0:
            print('{}/{}'.format(i, len(imgs)))
            
        midname = imgname[imgname.rindex("/")+1:]
        img = load_img(data_path +"/" + midname, color_mode = "grayscale")
        label = load_img(label_path +"/"+ midname, color_mode = "grayscale")
        img=img.resize((out_rows,out_cols))
        label=label.resize((out_rows,out_cols))

        img = img_to_array(img)
        label = img_to_array(label)
        imgdatas[i] = img
        imglabels[i] = label

    print('loading done')
    np.save(npy_path + '/{}.npy'.format(name), imgdatas)
    np.save(npy_path + '/{}_label.npy'.format(name), imglabels)
    print('Saving to .npy files done.')
    
def create_test_data(test_path, test_label_path, npy_path, out_rows, out_cols, name, img_type):

    print('-'*30)
    print('Creating test images...')
    print('-'*30)

    i = 0
    imgs = glob.glob(test_path+"/*."+img_type)
    imgdatas = np.ndarray((len(imgs),out_rows,out_cols,1), dtype=np.uint8)
    imglabels = np.ndarray((len(imgs),out_rows,out_cols,1), dtype=np.uint8)
    
    imgnames=[]
    for j, imgname in enumerate(imgs):
        if j%100==0:
            print('{}/{}'.format(j, len(imgs)))
        midname = imgname[imgname.rindex("/")+1:-4]
#         print(midname)
        img = load_img(test_path + "/" + midname+'.png', color_mode = "grayscale")
        label = load_img(test_label_path +"/"+ midname+'.png', color_mode = "grayscale")
        img=img.resize((out_rows,out_cols))
        label=label.resize((out_rows,out_cols))

        img = img_to_array(img)
        label = img_to_array(label)
        imgdatas[j] = img
        imglabels[j] = label
        imgnames.append(midname)
        
#     print(imgnames)    
    print('loading done')
    np.save(npy_path + '/{}.npy'.format(name), imgdatas)
    np.save(npy_path + '/{}_label.npy'.format(name), imglabels)
    np.save(npy_path + '/{}_name.npy'.format(name), imgnames)
    print('Saving to imgs_test.npy files done.')
    
def load_train_data(train_npy_path, mask_npy_path):
    print('-'*30)
    print('load train images...')
    print('-'*30)
    
    imgs_train = np.load(train_npy_path)
    imgs_mask_train = np.load(mask_npy_path)

    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')
    print('img : ', imgs_train.max())
    print('mask : ',imgs_mask_train.max())
    
    print('-'*30)
    print('normalization start...')
    print('-'*30)
    imgs_train = imgs_train/255.0
    
    imgs_mask_train[imgs_mask_train <= 127] = 0
    imgs_mask_train[imgs_mask_train > 127] = 1
    
    print('img : ',imgs_train.max())
    print('mask : ',imgs_mask_train.max())
    
    return imgs_train, imgs_mask_train

def load_test_data(test_npy_path):
    print('-'*30)
    print('load test images...')
    print('-'*30)
    
    imgs_test = np.load(test_npy_path)
    imgs_test = imgs_test.astype('float32')
    imgs_test /= 255
    
    return imgs_test

def load_data(train_npy_path, mask_npy_path, test_npy_path):
    imgs_train, imgs_mask_train = load_train_data(train_npy_path, mask_npy_path)
    imgs_test = load_test_data(test_npy_path)
    print(imgs_train.shape)
    print(imgs_mask_train.shape)
    print(imgs_test.shape)
    return (imgs_train, imgs_mask_train, imgs_test)