import csv
import cv2
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Dropout,Activation
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2
from keras.optimizers import Adam


####################Supporting Functions##################
def preprocess_img(img):
   # original shape: 160x320x3, input shape for neural net: 66x200x3
   # crop to 90x320x3
   process_img = img[50:140,:,:]
   # resize the image to 66x200x3 (same as nVidia)
   process_img = cv2.resize(process_img,(200, 66), interpolation = cv2.INTER_AREA)
   # convert to YUV color space (as nVidia paper suggests)
   process_img = cv2.cvtColor(process_img, cv2.COLOR_BGR2YUV)

   return process_img
####################Data Parsing##########################
steering = []
images = []
with open('/veh_data/driving_log.csv', newline='') as csvfile:
    lines = csv.DictReader(csvfile)
    for line in lines:
        #####################Center Image###########################
        filename = line['center'].split("/")[-1]
        path = "/veh_data/IMG/" + filename
        image = cv2.imread(path)
        
        '''
        Add Center Camera Data
        '''
        images.append(preprocess_img(image))
        steering.append(float(line['steering']))
        
        '''
        Augmentation : Center Camera Image Flipping
        '''
        images.append(preprocess_img(cv2.flip(image,1)))
        steering.append(-float(line['steering']))
        
        '''
        Augmentation: Brighter and Darker  Center Image
        '''
        gamma = 30
        images.append(preprocess_img(cv2.addWeighted(image, 1.0, numpy.zeros(image.shape,image.dtype),0.0,gamma)))
        steering.append(float(line['steering']))
        gamma = -30
        images.append(preprocess_img(cv2.addWeighted(image, 1.0, numpy.zeros(image.shape,image.dtype),0.0,gamma)))
        steering.append(float(line['steering']))
        
        #####################Left Image###########################
        correction = 0.2
        filename = line['left'].split("/")[-1]
        path = "/veh_data/IMG/" + filename
        image = cv2.imread(path)
        
        '''
        Augmentation : Left Camera adding
        '''
        images.append(preprocess_img(image))
        steering.append(float(line['steering']) + correction)
        
        '''
        Augmentation : Center Camera Image Flipping
        '''
        images.append(preprocess_img(cv2.flip(image,1)))
        steering.append(-float(line['steering']) + correction)
        
        '''
        Augmentation: Brighter and Darker  Left Image
        '''
        gamma = 30
        images.append(preprocess_img(cv2.addWeighted(image, 1.0, numpy.zeros(image.shape,image.dtype),0.0,gamma)))
        steering.append(float(line['steering']) + correction)
        gamma = -30
        images.append(preprocess_img(cv2.addWeighted(image, 1.0, numpy.zeros(image.shape,image.dtype),0.0,gamma)))
        steering.append(float(line['steering']) + correction)
        
        #####################Right Image###########################
        filename = line['right'].split("/")[-1]
        path = "/veh_data/IMG/" + filename
        image = cv2.imread(path)
        '''
        Augmentation : Right Camera adding
        '''
        images.append(preprocess_img(image))
        steering.append(float(line['steering']) - correction)
        
        '''
        Augmentation : Center Camera Image Flipping
        '''
        images.append(preprocess_img(cv2.flip(image,1)))
        steering.append(-float(line['steering']) - correction)
        
        '''
        Augmentation: Brighter and Darker  Right Image
        '''
        gamma = 50
        images.append(preprocess_img(cv2.addWeighted(image, 1.0, numpy.zeros(image.shape,image.dtype),0.0,gamma)))
        steering.append(float(line['steering']) - correction)
        gamma = -50
        images.append(preprocess_img(cv2.addWeighted(image, 1.0, numpy.zeros(image.shape,image.dtype),0.0,gamma)))
        steering.append(float(line['steering']) - correction)
        

################Data Analysis and Adjustment###############
c = list(zip(steering, images))
random.shuffle(c)
steering, images = zip(*c)

steering_rounded = [ round(steer,1) for steer in steering ]

init = float(-1.0)
hist = [init + ( float(i) / 10 ) for i  in range(0,21)]
hist = [ round(elm,1) for elm in hist ]

bars = [0] * len(hist)

for i in range(len(steering_rounded)):
    for j in range(len(hist)):
        if (steering_rounded[i] == hist[j]):
            bars[j] += 1
            continue

fig, ax = plt.subplots(1, 1)
plt.xticks(hist)
plt.bar(hist,bars,align = 'center',alpha = 0.5,width=0.02)
plt.savefig('/output/Data Dist Before Adjsutment.pdf')


counterMiddle = 0
counterLeft = 0
counterRight = 0
steering_flitered = []
images_filtered = []

for i in range(len(steering_rounded)-1,0,-1):
    if((steering_rounded[i] == 0.0 or steering_rounded[i] == -0.0) and counterMiddle <= 34740):
        counterMiddle +=1
        continue
    if(steering_rounded[i] == -0.2 and counterLeft <= 35922):
        counterLeft += 1
        continue
    if(steering_rounded[i] == 0.2 and counterRight <= 35922):
        counterRight += 1
        continue
    else:
        steering_flitered.append(steering[i])
        images_filtered.append(images[i])

steering_filtered_rounded = [ round(steer,1) for steer in steering_flitered ]

bars_filtered = [0] * len(hist)

for i in range(len(steering_filtered_rounded)):
    for j in range(len(hist)):
        if (steering_filtered_rounded[i] == hist[j]):
            bars_filtered[j] += 1
            continue

fig, ax = plt.subplots(1, 1)
plt.xticks(hist)
plt.bar(hist,bars_filtered,align = 'center',alpha = 0.5,width=0.02)
plt.savefig('/output/Data Dist After Adjsutment.pdf')
#######################Training Data#######################
X_train = numpy.array(images_filtered)
y_train = numpy.array(steering_flitered)
image_size = images[0].shape

######################Build Model#########################
print("Intializing Network ")
model = Sequential()
#Add PreProcessing Layer
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (66,200,3)))  

#Nvidia Model
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu', W_regularizer=l2(0.001)))
#model.add(Dropout(0.2))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu', W_regularizer=l2(0.001)))
#model.add(Dropout(0.2))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu', W_regularizer=l2(0.001)))
#model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3, activation='relu', W_regularizer=l2(0.001)))
#model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3, activation='relu', W_regularizer=l2(0.001)))
#model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation='relu', W_regularizer=l2(0.001)))
#model.add(Dropout(0.2))
model.add(Dense(50, activation='relu', W_regularizer=l2(0.001)))
#model.add(Dropout(0.2))
model.add(Dense(10, activation='relu', W_regularizer=l2(0.001)))
#model.add(Dropout(0.2))
model.add(Dense(1))
    
#Compile Model
model.compile(loss = 'mse', optimizer=Adam(lr=0.0001))
history = model.fit(X_train, y_train, batch_size=96 , nb_epoch=10, validation_split=0.2, shuffle='True', verbose=2)
model.save('/output/model.h5')

#########################Ploting MSE#################################
fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('/output/MSE.pdf')