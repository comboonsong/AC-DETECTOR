from utils import *

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

#Model architecture
import keras
from keras.engine.training import Model as KerasModel
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import MaxPooling2D, Input, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model, load_model

filepath = '/content/drive/MyDrive/ACdetector/data/nodif/nodif1.nii'
orgnifti = nib.load(filepath)
orgnifti = cleanNifti(orgnifti)
orgaffine, orgres = orgnifti.affine, orgnifti.header.get_zooms()
p1 = np.floor(np.array(orgnifti.header.get_data_shape())/2) #org_center

#set origin of orgnifti to center p1
invM = np.linalg.inv(orgaffine)
invM[:3,3] = p1
newaffine = np.linalg.inv(invM)
empty_header = nib.Nifti1Header()
newnifti = nib.Nifti1Image(orgnifti.get_fdata(), newaffine, empty_header) #create newnifti

#resampling
target_shape = np.array([60,60,60])
target_resolution = [-2,2,2]
target_affine = np.zeros((4,4))
target_affine[:3,:3] = np.diag(target_resolution)
target_affine[:3,3] = target_shape*target_resolution/2.*-1
target_affine[3,3] = 1.
resampnifti = resample_img(newnifti, target_affine=target_affine, target_shape=target_shape, interpolation='nearest')

#get information
resampdata = resampnifti.get_fdata()
DATATEST = []
DATATEST.append(resampdata)
DATATEST = np.stack(DATATEST,axis=0)

normDATATEST = normalizeData(DATATEST)

modelpath = 'experiment5-nodifAD-resamp60res-inpFULLSTACK-lr1e4-10000epch.h5'
model = load_model(modelpath)
predX = model.predict(normDATATEST) #predict in downsampling space

#change prediction to upsampling space
downsampres = resampnifti.header.get_zooms()
upsampres = newnifti.header.get_zooms()
upsamp_center = p1
downsamp_center = np.linalg.inv(target_affine)[:3,3]
downsamp_origin = predX
points = (downsamp_center, downsamp_origin, upsamp_center)
upsamp_origin =  mapPoint(np.array(downsampres),np.array(upsampres),points)

print(upsamp_origin)

