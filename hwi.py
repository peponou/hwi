!pip install lap
# Read the dataset description
import gzip
# Read or generate p2h, a dictionary of image name to image id (picture to hash)
import pickle
import platform
import random
# Suppress annoying stderr output when importing keras.
import sys
from lap import lapjv
from math import sqrt
# Determine the size of each image
from os.path import isfile

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as pil_image
from imagehash import phash
from keras import backend as K
from keras import regularizers
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, \
    Lambda, MaxPooling2D, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import Sequence
from pandas import read_csv
from scipy.ndimage import affine_transform
from tqdm import tqdm_notebook as tqdm
import time
import shutil
import os
import sys
import imageio
from glob import glob
import csv
from distutils.dir_util import copy_tree






TRAIN_DF = "../input/humpback-whale-identification/train.csv"
SUB_DF = "../input/humpback-whale-identification/sample_submission.csv"
TRAIN = '../input/humpback-whale-identification/train/'
TEST = '../input/humpback-whale-identification/test/'
P2H = '../input/metadata/p2h.pickle'
P2SIZE = '../input/metadata/p2size.pickle'
BB_DF = "../input/metadata/bounding_boxes.csv"
tagged = dict([(p, w) for _, p, w in read_csv(TRAIN_DF).to_records()])
submit = [p for _, p, _ in read_csv(SUB_DF).to_records()]
join = list(tagged.keys()) + submit






# shutil.rmtree('/kaggle/working/')






import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
from time import time
df = pd.read_csv(TRAIN_DF)
df.head()
bbox_df = pd.read_csv(BB_DF)
bbox_df.head()
df = df.merge(bbox_df, on=['Image'])
def crop_loose_bbox(img,area, val=0.2):
    img_w, img_h = img.size
    w = area[2] - area[0]
    h = area[3] - area[1]
    area2 = (max(0, int(area[0] - 0.5*val*w)),
             max(0, int(area[1] - 0.5*val*h)),
             min(img_w, int(area[2] + 0.5*val*w)),
             min(img_h, int(area[3] + 0.5*val*h)))
    return img.crop(area2)
try:
    os.makedirs('/kaggle/working/crop_train')
    os.makedirs('/kaggle/working/crop_test')
except:
    pass
    
t=time()
for i in range(len(df)):
    fn = os.path.join(TRAIN, df.Image[i])
    img = Image.open(fn)
    area = (df.x0[i],df.y0[i],df.x1[i],df.y1[i])
    cropped_img = crop_loose_bbox(img,area, 0.2)
    cropped_img.save(fn.replace(TRAIN, '/kaggle/working/crop_train/'))
    if i %3000 == 0:
        print(i)
        print(time() -t , 'sec')
print(time() -t )
df = pd.read_csv(SUB_DF)
df.head()
bbox_df = pd.read_csv(BB_DF)
bbox_df.head()
df = df.merge(bbox_df, on=['Image'])

t=time()
for i in range(len(df)):
    fn = os.path.join(TEST, df.Image[i])
    img = Image.open(fn)
    area = (df.x0[i],df.y0[i],df.x1[i],df.y1[i])
    cropped_img = crop_loose_bbox(img,area, 0.2)
    cropped_img.save(fn.replace(TEST, '/kaggle/working/crop_test/'))
    if i %3000 == 0:
        print(i)
        print(time() -t , 'sec')
print(time() -t )






# removing new_whale
TRAIN_NO_NEW_WHALE_DF = os.path.join(os.getcwd(), "train_without_new_whale.csv")

with open(TRAIN_DF, "r") as inp, open(TRAIN_NO_NEW_WHALE_DF, 'w') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[1] != "new_whale":
            writer.writerow(row)






# also applying grayscale to images
df = pd.read_csv(TRAIN_NO_NEW_WHALE_DF)
df.head()

try:
    os.makedirs('/kaggle/working/crop_test_grayscale')
    os.makedirs('/kaggle/working/crop_train_without_new_whale')
except:
    pass
    
t=time()
for i in range(len(df)):
    fn = os.path.join('/kaggle/working/crop_train/', df.Image[i])
    img = Image.open(fn).convert('L')
    img.save(fn.replace('/kaggle/working/crop_train/', '/kaggle/working/crop_train_without_new_whale/'))
    if i %3000 == 0:
        print(i)
        print(time() -t , 'sec')
print(time() -t )
df = pd.read_csv(SUB_DF)
df.head()
bbox_df = pd.read_csv(BB_DF)
bbox_df.head()
df = df.merge(bbox_df, on=['Image'])

t=time()
for i in range(len(df)):
    fn = os.path.join('/kaggle/working/crop_test/', df.Image[i])
    img = Image.open(fn).convert('L')
    img.save(fn.replace('/kaggle/working/crop_test/', '/kaggle/working/crop_test_grayscale/'))
    if i %3000 == 0:
        print(i)
        print(time() -t , 'sec')
print(time() -t )






print(len(os.listdir(os.path.join(os.getcwd(), "crop_train_without_new_whale"))))
print(len(os.listdir(os.path.join(os.getcwd(), "crop_test_grayscale"))))




# importing the libraries for model training
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.applications import VGG16


IMAGE_SIZE = [224, 224]   

# loading the weights of VGG16 without the top layer. These weights are trained on Imagenet dataset.
# input_shape = (64,64,3) as
vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)  

# this will exclude the initial layers from training phase as there are already been trained.
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
x = Dense(128, activation = 'relu')(x)  
x = Dense(5004, activation = 'softmax')(x)  # adding the output layer with softmax function as this is a multi label classification problem.

model = Model(inputs = vgg.input, outputs = x)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])






# importing the libraries image Augmentation
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

from shutil import copyfile
copyfile(SUB_DF, "/kaggle/working/sample_submission_edit_1.csv")
# SUB_EDIT1_DF = "/kaggle/working/train1/"
# datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

datagen = ImageDataGenerator(
                            rescale=1./255,   # all pixel values will be between 0 an 1
                            shear_range=0.2, 
                            zoom_range=0.2,
                            horizontal_flip=True,
                            preprocessing_function=preprocess_input, validation_split=0.2)

# validation_datagen = ImageDataGenerator(rescale = 1./255, preprocessing_function=preprocess_input, validation_split=0.1)

traindf=pd.read_csv(TRAIN_NO_NEW_WHALE_DF,dtype=str)
testdf=pd.read_csv("/kaggle/working/sample_submission_edit_1.csv",dtype=str)
IMAGE_SIZE = [224, 224]   

train_generator=datagen.flow_from_dataframe(dataframe=traindf,
                            directory=os.path.join(os.getcwd(), "crop_train_without_new_whale"), 
                            x_col="Image", y_col="Id", subset="training", batch_size=200,
                            shuffle=True, class_mode="categorical", target_size=IMAGE_SIZE)

valid_generator=datagen.flow_from_dataframe(dataframe=traindf,
                            directory=os.path.join(os.getcwd(), "crop_train_without_new_whale"), 
                            x_col="Image", y_col="Id", subset="validation", batch_size=200,
                            shuffle=True, class_mode="categorical", target_size=IMAGE_SIZE) 


test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe( dataframe=testdf,
                            directory=os.path.join(os.getcwd(), "crop_test_grayscale"), 
                            x_col="Image", y_col=None, batch_size=200, 
                            shuffle=False, class_mode=None, target_size=IMAGE_SIZE)





csvfile = open(TRAIN_NO_NEW_WHALE_DF, 'r').readlines()
filename = "training_set"
open(str(filename) + '.csv', 'w+').writelines(csvfile[0:12558])
filename1 = "validation_set"
open(str(filename1) + '.csv', 'w+').writelines(csvfile[12558:(len(csvfile)+1)])




print(len(csvfile))
print(len(open("/kaggle/working/training_set.csv", 'r').readlines()))
print(len(open("/kaggle/working/validation_set.csv", 'r').readlines()))





STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
print(STEP_SIZE_TRAIN)
print(STEP_SIZE_VALID)
print(STEP_SIZE_TEST)
model.fit_generator(generator=train_generator,
                    steps_per_epoch=500,
                    validation_data=valid_generator,
                    validation_steps=150,
                    epochs=10
)





model.evaluate_generator(generator=valid_generator, steps=15, verbose=1)







test_generator.reset()
pred=model.predict_generator(test_generator, steps=40, verbose=1)






predicted_class_indices=np.argsort(pred)[:, ::-1][:, :5]





labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k,i] for k,i in predicted_class_indices]






def top_5_pred_labels(classes):
    
    labels = []
    for i in range(predicted_class_indices.shape[0]):
        labels.append(' '.join([classes[idx] for idx in predicted_class_indices[i]]))
    return labels
top_5_pred_labels(labels)




filenames=test_generator.filenames
results=pd.DataFrame({"Image":filenames,
                      "Id":top_5_pred_labels(labels)})
results.to_csv("submission.csv",index=False)


(pd.read_csv("/kaggle/working/results.csv",dtype=str)).head(8000)