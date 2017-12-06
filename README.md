#**Behavioral Cloning**
---
[//]: # (Image References)

[image1]: ./examples/nvidia_model.png "Nvidia Model"
[image2]: ./examples/MSE.png "MSE"
[image3]: ./examples/recover_2.jpg "Recovery Image"
[image4]: ./examples/recover_3.jpg "Recovery Image"
[image5]: ./examples/recover_4.jpg "Recovery Image"
[image6]: ./examples/recover_5.jpg "Recovery Image"
[image7]: ./examples/my_model.jpg "My Model"
[image8]: ./examples/Dist_bef.png "Dist Before"
[image9]: ./examples/Dist_aft.png "Dist After"

##Overview

The project main idea is to use deep neural networks and constitutional neural networks to clone driving behavior.

##Project Goals
The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality


####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* **model.py** containing the script to create and train the model
* **drive.py** for driving the car in autonomous mode
* **model.h5** containing a trained convolution neural network 
* **README.md** summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my **drive.py** file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The **model.py** file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is a modified version of [Nvidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) used for end to end deep learning. The model is defined in (**model.py** lines 181-203)

This is the original model of Nvidia

![alt text][image1]

####2. Attempts to reduce overfitting in the model

In order to reduce over fitting two approaches where applied:

* Regularization

	L2 Reguralization with value (0.001) is used as it had the best performance in reducing overfitting.

* Data Augmentation

	1) Adding flipped center images, left and right camera images, and blurred images replica where used in order to reduce overfitting.

	2) Clockwise lap was performed so that the car would be able to see different view of the environmt.

	3) Track 2 lap was performed to make the data diverse and not overfit on one track

![alt text][image2]

####3. Model parameter tuning

(**model.py** line 206-207)

* Learning Rate : 0.0001
* Epochs : 10 epochs
* Optimizer : Adam Optimizer (so the learning rate was not tuned manually)

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of:

Track 1

* Center lane driving
* Recovering from the left and right sides of the road 
* Smooth drives around curves
* One lap clock-wise
* Blurred Images(Brighter and Darker)

Track 2

* Left lane driving
* right lane driving
* Smooth drives around curves
* One lap clock-wise
* Blurred Images(Brighter and Darker)

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

The overall strategy for deriving a model architecture was to make the vehicle able to drive around the track in autonomous mode without leaving the track.

My first step was to use a convolution neural network model similar to the Nvidia I thought this model might be appropriate because it was used by Nvidia in its end to end autonomous car project therefore its a suitable choice for the current target.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was over fitting. 

To combat the overfitting, I added L2 Regularization which proved to be the best regularization approach.I had a trial using Dropout but the results was not good.

Then I've preprocessed the images to fit the model as the original image (160x320x3) and Nvidia image (66x200x3). So it is recommended to resize the input images for the model.

Also i have added data augmentation from both track 1 and track 2 as follows:

Track 1

* Center lane driving
* Recovering from the left and right sides of the road 
* Smooth drives around curves
* One lap clock-wise
* Blurred Images(Brighter and Darker)

Track 2

* Left lane driving
* right lane driving
* Smooth drives around curves
* One lap clock-wise
* Blurred Images(Brighter and Darker)

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track due to loss of control and not being able to recover fast enough so to improve the driving behavior in these cases, I created a recovery lap where i get the vehicle in a condition of going of the road then i start recording the recovery action from this state so the vehicle would be able to recover from those situations autonomously.

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

Finally, 
At the end of the process, the vehicle is able to drive autonomously around both tracks without leaving the road.


####2. Final Model Architecture

The final model architecture (**model.py** lines 180-203)

Architecture visualization (inspired by the architecture used by jeremy shannon)

![alt text][image7]


####3. Creation of the Training Set & Training Process

#####3.1 Data Collection

I collected data from both tracks 1 and 2 to make the data more diverse and the network in turn would be more generic.

The collected data is as follows:

Track 1

* Center lane driving
* Recovering from the left and right sides of the road 
* Smooth drives around curves
* One lap clock-wise
* Blurred Images(Brighter and Darker)

Track 2

* Left lane driving
* right lane driving
* Smooth drives around curves
* One lap clock-wise
* Blurred Images(Brighter and Darker)

#####3.2 Data Augmentation

* Adding both left and right cameras input with steering adjustment factor of (+/-) ***0.2*** 
* Adding bright and dark blurred augmentation for all the images (including adjusted left and right images)

#####3.3 Data PreProcessing

I decided to change in the images to be like the ones used by Nvidia in their model to be able to use the model perfectly so I had to do the following :

* Cropping images 50 pixels from the top and 20 pixels from the bottom) so the image size converted from 160x320x3 to 90x320x3

* Convert images from BGR to YUV as cv2.imread() reads the images in BGR
* Re-size the image from 90x320x3 to 66x200x3

The last three operations have been done in **model.py** and in **drive.py** although in **drive.py** the second operation was to convert from RGB to YUV

#####3.4 Data Analysis and Adjustment

After Analyzing the data which would be fed to the network i have found that alot of data is given for driving in a straight line also alot of data acquired from left and right camera causing bias in the data for both ***0.0*** and ***(+/-) 0.2*** steering angles.

![alt text][image8]

The Solution was to trim the data to end with a normally distributed data for all the angles which would result in having the network learn better all angles.

![alt text][image9]

#### Upcoming Improvements

I used FloydHub GPU and I saved the preprocessed data online therefore i didnt use generators but i intend to enhance this project furthermore"# CarND_Behavioural_Clonning_P3" 
