from email.utils import collapse_rfc2231_value
import nibabel as nib
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
from nilearn.image import resample_img
from monai.transforms import (
    LoadImage,
    Affine,
    Affined
)
from monai.config import print_config
from monai.apps import download_and_extract

def removeNan(nifti):
    data = nifti.get_fdata()
    data[np.isnan(data)] = 0 #check whether an intensity is nan
    return data

def cleanNifti(nifti):
    data = removeNan(nifti)
    newNifti = nib.Nifti1Image(data, nifti.affine, nifti.header)
    return newNifti

def selectAxial(data, z):
    return data[:,:,z]

def selectSagital(data, x):
    return data[x,:,:]

def selectCoronal(data, y):
    return data[:,y,:]

def rotate3D(data_dict, rotate_params):
    Rotate = Affined(
        keys=['image','label'],
        rotate_params=rotate_params,
        spatial_size=(90,90,90),
        padding_mode="zeros",
        mode=("bilinear", "nearest")
        )

    rotated_data_dict = Rotate(data_dict)
    return rotated_data_dict

#Min-Max Scaling
def normalizeData(X):
    X_normalized = cv2.normalize(X, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    return X_normalized

def mapPoint(org_res,new_res,points):
    p1, p2, q1 = points #(org_center, org_origin, new_center)
    org_pixel_vec = p2-p1
    distance_vec = org_pixel_vec * org_res
    new_pixel_vec = distance_vec / new_res
    q2 = new_pixel_vec + q1
    return q2

#label
def loadLabel(csvpath):
    df_label = pd.read_csv(csvpath, header=None)
    df_label.columns = ['niftipath','x','y','z']
    return df_label


#show plot image
#show train slice image
def showPred(DATA, label, pred, size):
    mark = DATA.copy()
    mark2 = DATA.copy()
    mark[label[0]-size:label[0]+size,label[1]-size:label[1]+size] = np.max(DATA)
    mark2[label[0]-size:label[0]+size,label[1]-size:label[1]+size] = np.max(DATA)
    mark2[pred[0]-size:pred[0]+size,pred[1]-size:pred[1]+size] = np.max(DATA)
    show_fig = np.concatenate((mark,mark2),axis=1)
    plt.figure(figsize=(20, 20))
    plt.imshow(show_fig)

def deg2PI(deg):
    Pi = deg * np.pi / 180
    return Pi

def drawImage(imgdata, slice_point, label=None, save=None):

    def drawLabel(imgdata_2d, label, side=1):
        imgdata_mark = imgdata_2d.copy()
        imgdata_mark[label[0]-side:label[0]+side,label[1]-side,label[1]+side] = np.max(imgdata_2d)
        return imgdata_mark

    x,y,z = slice_point
    if not label:
        axial_org = imgdata[:,:,z]
        sagital_org = imgdata[x,:,:]
        coronal_org = imgdata[:,y,:]
        show_fig = np.concatenate((axial_org,sagital_org,coronal_org),axis=1)
        figure = plt.figure(figsize=(10,10))
        plt.imshow(show_fig)
        
    else:
        if len(label) == 1:
            axial_org = imgdata[:,:,z]
            axial_mark = drawLabel(axial_org, label[0][:2])
            sagital_org = imgdata[x,:,:]
            sagital_mark = drawLabel(sagital_org, label[0][1:])
            coronal_org = imgdata[:,y,:]
            coronal_mark = drawLabel(coronal_org, label[[0,2]])
            show_fig = np.concatenate((axial_mark,sagital_mark,coronal_mark),axis=1)
            plt.imshow(show_fig)

        elif len(label) == 2:
            axial_org = imgdata[:,:,z]
            axial_mark = drawLabel(axial_org, label[0][:2])
            axial_mark2 = drawLabel(axial_org, label[1][:2])
            
            sagital_org = imgdata[x,:,:]
            sagital_mark = drawLabel(sagital_org, label[0][1:])
            sagital_mark2 = drawLabel(sagital_org, label[1][:2])

            coronal_org = imgdata[:,y,:]
            coronal_mark = drawLabel(coronal_org, label[[0,2]])
            coronal_mark2 = drawLabel(coronal_org, label[1][:2])

            col1 = np.concatenate((axial_mark,axial_mark2),axis=0)
            col2 = np.concatenate((sagital_mark,sagital_mark2),axis=0)
            col3 = np.concatenate((coronal_mark,coronal_mark2),axis=0)
            show_fig = np.concatenate((col1,col2,col3),axis=1)
            plt.imshow(show_fig)
            
        else:
            raise Exception("Label is wrong.")




