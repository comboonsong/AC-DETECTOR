from utils import *
dir='/content/drive/MyDrive/ACdetector/data/nodif-AD/'
Filename = [os.path.join(dir,'nodif_'+'0'*(4-len(str(10)))+str(10)+'.nii') for i in range(380)]

#label
csvpath = '/content/drive/MyDrive/ACdetector/data/label/nodif-AD_label.csv'
df_label = loadLabel(csvpath)

#load images(nifti)
#stack image and label_img and label_text simultaneously
target_shape = np.array([90,90,90])
Data = []
LABEL_img = []
LABEL_txt = []

#stack all images in a folder
for i in range(len(df_label)):
    filepath = Filename[i]
    label = df_label.iloc[i][['x','y','z']].values
    label = np.array([label[0],label[1],label[2]])
    orgnifti = nib.load(filepath)
    orgnifti = cleanNifti(orgnifti)
    orgaffine, orgres = orgnifti.affine, orgnifti.header.get_zooms()
    p1 = np.floor(np.array(orgnifti.header.get_data_shape())/2) #org_center
    p2 = label #org_origin
    
    #resampling
    target_resolution = [-2,2,2]
    target_affine = np.zeros((4,4))
    target_affine[:3,:3] = np.diag(target_resolution)
    target_affine[:3,3] = target_shape*target_resolution/2.*-1
    target_affine[3,3] = 1.
    resampnifti = resample_img(orgnifti, target_affine=target_affine, target_shape=target_shape, interpolation='nearest')
    
    #get new label
    resampres = resampnifti.header.get_zooms()
    q1 = np.abs(np.linalg.inv(target_affine)[:3,3])
    newlabel = np.round(mapPoint(np.array(orgres),np.array(resampres),points=(p1,p2,q1)))
    newlabel = [int(newlabel[0]),int(newlabel[1]),int(newlabel[2])]

    resampdata = resampnifti.get_fdata()
    Data.append(resampdata)
    LABEL_txt.append(newlabel)
    label_img = np.zeros(target_shape, dtype='uint8')
    label_img[newlabel[0],newlabel[1],newlabel[2]] = 255
    LABEL_img.append(label_img)
    print(f'at {i}')

Data = np.stack(Data,axis=0)
LABEL_img = np.stack(LABEL_img,axis=0)
Data_dict = {'image':Data, 'label':LABEL_img}

#augmentation by rotating
#x=-5,+5, y=-5,+5, z=-5+5
angle_range = [-deg2PI(i) for i in [0,-5,5]]
Rotate_params = [(r1,r2,r3) for r1 in angle_range for r2 in angle_range for r3 in angle_range]

rotated_data_dict = rotate3D(data_dict=Data_dict, rotate_params=Rotate_params[0])
rotated_data = rotated_data_dict['image']
X = np.argwhere(rotated_data_dict['label'] != 0)

rotated_data_dict = rotate3D(data_dict=Data_dict, rotate_params=Rotate_params[0])
X = np.argwhere(rotated_data_dict['label'] != 0)
rotated_data = rotated_data_dict['image']
rotated_data = rotated_data[X[:,0]] 

for i,r in enumerate(Rotate_params[1:11]):
    print('at', i+1)
    rotated_data_dict = rotate3D(data_dict=Data_dict, rotate_params=r)
    X0 = np.argwhere(rotated_data_dict['label'] != 0)
    X = np.append(X, X0, axis=0)
    rotated_data = np.append(rotated_data, rotated_data_dict['image'][X0[:,0]], axis=0)

#min-max scaling normalization
DATA = normalizeData(rotated_data)
LABEL = X[:,1:]

from sklearn.model_selection import train_test_split
split = train_test_split(DATA, LABEL, test_size=0.10,random_state=42)
(trainImages, testImages), (trainTargets, testTargets) = split[:2], split[2:4]

def build_model():
    
    INPUT_HEIGHT = 60
    INPUT_WIDTH = 60
    
    # test with two branches of conv, 2 pts only
    MOMENTUM = 0.5
    if K.image_data_format() == 'channels_first':
        inp = Input(shape=(60, INPUT_HEIGHT, INPUT_WIDTH), name='input')
    elif K.image_data_format() == 'channels_last':
        inp = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 60), name='input')
    else:
        print('image_data_format at build_model error!')
        return
 
    x = Conv2D(32, 3, 3, padding='same')(inp)
    x = BatchNormalization(momentum=MOMENTUM)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Dropout(0.1)(x)
   
    x = Conv2D(64, 3, 3, padding='same')(x)
    x = BatchNormalization(momentum=MOMENTUM)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Dropout(0.2)(x)
 
    x = Conv2D(96, 1, 1, padding='same')(x)
    x = BatchNormalization(momentum=MOMENTUM)(x)
    x = Conv2D(128, 3, 3, padding='same')(x)
    x = BatchNormalization(momentum=MOMENTUM)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Dropout(0.3)(x)

    #for point
    x = Conv2D(192, 1, 1, padding='same')(x)
    x = BatchNormalization(momentum=MOMENTUM)(x)
    x = Conv2D(256, 3, 3, padding='same')(x)
    x = BatchNormalization(momentum=MOMENTUM)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Dropout(0.3)(x)
 
    x = Flatten()(x)
 
    x = Dense(768)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    outp_point = Dense(3, activation='relu', name = 'outp_p')(x)
      
    model = KerasModel(inputs=inp, outputs=outp_point)
 

    model.compile(loss ='mse',
              optimizer = Adam(learning_rate = 1e-4),
              metrics =['accuracy'])
 
 
    model.summary()
 
    return model

BATCH = 32
EPOCHS = 2000

H = model.fit(
    trainImages, trainTargets,
    validation_data=(testImages, testTargets),
    batch_size=BATCH,
    epochs=EPOCHS,
    verbose=1)

modelpath = 'experiment5-nodifAD-resamp60res-inpFULLSTACK-chanelSAG-lr1e4-2000epch.h5'
model.save(modelpath, save_format="h5")