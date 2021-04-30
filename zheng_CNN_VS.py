#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 09:33:13 2020

@author: lundmc
"""


"""
Hi ERAU team!

Below are the sections of code for the Variable Stride class, called MyLayer,
and its support function, CalcVSOutShape. I've left some comments in the codes, 
which may or may not be useful in helping you decipher what we're doing. 

There is good news and bad news- the bad news is that Python doesn't even make
my top 3 as far as preferred languages, so there is a lot of room for 
improvement in these codes. The good news is that they work, and you can 
accomplish a lot with this project, even if you never fully understand the 
codes.


As far as usage: 
    You will discover in your research of CNNs that the general structure for
creating a network in Keras/ Tensorflow is, depending on how you import 
packages, something along the lines of:
    
    model=Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    ... and so on
    
    
I have been adding in the Variable Stride layer at the very beginning of my 
network, so mine have started out with something like this:
    
    model=Sequential()
    model.add(MyLayer(self.input_shape, input_shape = self.input_shape))
    image_size=[int(model.output.shape[1]), int(model.output.shape[2])]
    model.add(Conv2D(32, (3, 3), activation='relu'))
    image_size=[int(model.output.shape[1]), int(model.output.shape[2])]
    ... and so on

I track the input and output shapes as a sanity check, which I find useful, but 
is not necessary. I also allow input_shape vary so I can use it with  different 
data sets, but you can hard code this for the retinopathy data.

Below is a list of packages I import and use to contstruct my network; you can 
see from the warnings that not all of them are used in the Variable Stride part 
of the codes. This is just a starting place. You will probably use different/ 
additional packages in the networks you create (hint: there are some good ones
out there for importing data and splitting it into training and testing sets).

Good luck!
Maggie Lund

"""



import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Flatten, Layer
from keras.optimizers import Adam
import keras.callbacks
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os, os.path
from keras import activations
from keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
import random 
from self import self 
from keras_visualizer import visualizer 
import graphviz

def CalcVSOutShape(input_shape,
                   step_v = [3, 1, 3],
                   step_h = [1, 5],
                   frac_v = [1/3, 1/3, 1/3],
                   frac_h = [1/4, 3/4]):
    print('hhh', input_shape)
    (nx, ny,nz) = input_shape
    ## Compute the cumulative fractions of rows(v) / columns(h)
    cum_frac_v = np.cumsum(frac_v)
    cum_frac_h = np.cumsum(frac_h)
    
    #### Estimate the output shape of the image
    ## #?# the algorithm shifts after exceeding the threshold
    frac_bounds_v = cum_frac_v * (nx-0) - 1
    frac_bounds_h = cum_frac_h * (ny-0) - 1
    
    out_h = 0
    out_v = 0
    
    idx_low = 0

    for frac_idx in range(len(frac_h)):
        
        idx_span = min(frac_bounds_h[frac_idx], ny-3) - idx_low
        
        divisor_h, remainder_h = divmod(idx_span, step_h[frac_idx])

        out_h += int(divisor_h) + 1
        idx_low += (int(divisor_h) + 1) * step_h[frac_idx]
    
    idx_low = 0
    
    for frac_idx in range(len(frac_v)):
        idx_span = min(frac_bounds_v[frac_idx], nx-3) - idx_low
        
        divisor_v, remainder_v = divmod(idx_span, step_v[frac_idx])
    
        out_v += int(divisor_v) + 1
        idx_low += (int(divisor_v) + 1) * step_v[frac_idx]
        
    return (out_v, out_h, nz)




class MyLayer(Layer):

    def __init__(self, 
                 _input_shape,
                 k1 =  np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]),
                 step_v = [3, 1, 3],
                 step_h = [1, 5],
                 frac_v = [1/3, 1/3, 1/3],
                 frac_h = [1/4, 3/4],
                 **kwargs): #output_dim?
        
        self._input_shape = _input_shape
        self.k1 = k1
        self.step_v = step_v
        self.step_h = step_h
        self.frac_v = frac_v
        self.frac_h = frac_h
        self.output_dim = CalcVSOutShape(_input_shape,
                   step_v = step_v,
                   step_h = step_h,
                   frac_v = frac_v,
                   frac_h = frac_h)
        
        super(MyLayer, self).__init__(**kwargs)

    ## needs output_dim and input_shape
        
    ## Might be done, do I need to stack/concatenate the results?
    def call(self, inputs):

        myimg = inputs #[0,:,:,:]
    
        (whatisthis,nx,ny,nz) = myimg.shape

        nx = int(nx)
        ny = int(ny)
        nz = int(nz)
        
        out_shape = CalcVSOutShape((nx, ny, nz),
                       step_v = self.step_v,
                       step_h = self.step_h,
                       frac_v = self.frac_v,
                       frac_h = self.frac_h)
        
        #print('predicted output shape', out_shape)

        try:
            batch_quant = int(whatisthis)
        except:
            
            print('placeholder tensor passed to call, pushing it forward in the desired output shape')
            newimg = myimg[:,:out_shape[0],:out_shape[1],:out_shape[2]]
            #print('77', newimg, newimg.shape, type(newimg))
            return newimg
        
        newimg = np.zeros((batch_quant,) + out_shape)    

        kernel = np.repeat(self.k1[:, :, np.newaxis], nz, axis=2)
    
        mystep_v = 0
        mystep_h = 0
        myfrac_v = 0
        myfrac_h = 0
        
        newrow = 0
        oldrow = 0
        oldcol = 0
        
        new_col_idx = 0
        new_row_idx = 0
        while oldrow < (nx-2):   
    
            while oldcol < (ny-2):  
                temp = tf.keras.backend.get_value(myimg[0,oldrow:oldrow+3,oldcol:oldcol+3,:])
   
                temp *= kernel

                newimg[new_row_idx, new_col_idx,:] =  np.sum(np.sum(temp,axis=0),axis=0)

                if oldcol > (np.sum(self.frac_h[0:myfrac_h+1]) * nx - 1):
                    
                    mystep_h = mystep_h + 1  #?# mystep_h always equals myfrac_h
                    myfrac_h = myfrac_h + 1
                    
                oldcol = oldcol + self.step_h[mystep_h]
                
                new_col_idx += 1
                
            if oldrow > (np.sum(self.frac_v[0:myfrac_v+1]) * ny - 1):
                    
                mystep_v = mystep_v + 1  #?# mystep_v always equals myfrac_v
                myfrac_v = myfrac_v + 1 
                
            oldcol = 0
            mystep_h = 0
            myfrac_h = 0
            newrow = newrow + 1
            oldrow = oldrow + self.step_v[mystep_v]
            new_row_idx += 1
    
        return K.constant(newimg)

    ## DONE
    def compute_output_shape(self, input_shape):
        #assert isinstance(input_shape, list)
        
        working_shape_tuple = input_shape[1:]
        
        new_shape = CalcVSOutShape(working_shape_tuple,
                                   step_v = self.step_v,
                                   step_h = self.step_h,
                                   frac_v = self.frac_v,
                                   frac_h = self.frac_h)
        
        return (input_shape[0],) + tuple(new_shape)  # pre-pend the batch_size :: input_shape[0]




num_class = 2

train = ImageDataGenerator(rescale = 1/255)
train_dataset = train.flow_from_directory('c:\\temp\\MA395\\train',
                                          target_size= (600,600),
                                          batch_size= 3,
                                          class_mode= 'binary')

validation_dataset = train.flow_from_directory('c:\\temp\\MA395\\validation',
                                          target_size= (600,600),
                                          batch_size= 3,
                                          class_mode= 'binary')

test_dataset = train.flow_from_directory('c:\\temp\\MA395\\test',
                                          target_size= (600,600),
                                          batch_size= 3,
                                          class_mode= 'binary')

model = Sequential()



model.add(Conv2D(32,(7,7), input_shape =(600,600,3)))
model.add(Activation('relu')) 

image_size=[int(model.output.shape[1]), int(model.output.shape[2]), int(model.output.shape[3])]
model.add(MyLayer(_input_shape = image_size))

    
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))

image_size=[int(model.output.shape[1]), int(model.output.shape[2]), int(model.output.shape[3])]
model.add(MyLayer(_input_shape = image_size))

model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))

image_size=[int(model.output.shape[1]), int(model.output.shape[2]), int(model.output.shape[3])]
model.add(MyLayer(_input_shape = image_size))

model.add(Flatten()) #turns to 1D
model.add(Dense(64)) #needs to be 1D
model.add(Activation('relu'))
#output layer
model.add(Dense(1))
model.add(Activation('sigmoid'))
                        
                                   



model.compile(loss= 'binary_crossentropy',
              optimizer = RMSprop(lr=.001),
              metrics =['accuracy'])

model_fit = model.fit(train_dataset,
                      steps_per_epoch = 3,
                      epochs= 50,
                      validation_data= validation_dataset)


trCATEGORIES = ["class1", "class4"]
dir_path = 'C:\\temp\\MA395\\test'

for category in trCATEGORIES:
    path = os.path.join(dir_path, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        
        plt.imshow(img_array, cmap='gray')
        plt.show()
        
        X = image.img_to_array(img_array)
        X = np.expand_dims(X,axis = 0)
        images = np.vstack([X])
        val = model.predict(images)
        if val == 0:
            print('class 1')
        else:
            print('class 4')   
            
model.summary()
score = model.evaluate(test_dataset, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
