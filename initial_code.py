#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 09:33:13 2020


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
