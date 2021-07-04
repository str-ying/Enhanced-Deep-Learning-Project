# Enhanded-Deep-Learning-Project

 

INTRODUCTION 
Nevada National Security Site (NNSS) has partnered with Embry Riddle Aeronautical University (ERAU) to collaborate in a research project regarding a novel pooling method that the NNSS has developed: variable stride. The NNSS is a high security support site to national laboratories. The NNSS conducts research and experiments with nuclear weapons, national security, and various environmental programs. They have a multi-faceted mission, a part of it being the Stewardship Science Experimentation (SSE). The SSE is centered around the National Nuclear Security Administration (NNSA) which does various non nuclear experiments. The NNSS hosts one of the SSE’s primary labs, the U1a underground complex which experiments for the nation’s subcritical experimentation program. A form of data collecting at this lab is through x-rays machines which produce images during experiments. The lab uses image processing methods to analyze the data and one method of image processing are convolutional neural networks. The NNSS has developed a new pooling method called variable stride. This project is supported by the National Science Foundation (NSF) through REU award number DMS - 2050754. 
 
PROJECT SCOPE
 The main goal of this project is to examine variable stride. Since it has been developed recently, it has not been used as much as the other pooling methods. To examine this method, a small data set of diabetic retinopathy eye images were given to use to implement into the convolutional neural network. Variable stride will be evaluated using metrics such as F-1 score and confusion matrix. Along with the examination, variable stride will be compared to AvgPool and MaxPool.
 
BACKGROUND
Neural Networks 
This project uses convolutional neural networks (CNN). CNNs are deep neural networks that can identify specific aspects of an image when trained. Neural networks are inspired by the human neural system. Each neuron is interconnected with each other to form a network. The neurons or nodes each have a weight which determines the importance of the neuron. The neuron is multiplied by its weight and summed up and passed into an activation function that calculates the output of the neuron. This process repeats for each layer and is called forward propagation. The network also does back propagation which takes the error rate of the output of the previous epoch and finetunes the weights using partial derivatives. Back propagation is essential for neural network training due to its fine tuning of the weights. The distinguishing factor of a CNN is its convolutional layer and pooling layer.


Convolutional Neural Network Layers
Within a CNN there are multiple layers that work to help identify the image: convolutional layer, pooling layer, and fully connected layer. The convolutional layer applies filters or weights that have values to help the machine detect features. The filters have stride and size that lets it know how many “steps” to take within the picture. The filters themselves are matrices that yield one number when applied to the image. Each convolutional layer is followed by an activation function. The pooling layer downsizes data to reduce the size of the matrix. The fully connected layer looks at the last convolutional layer and determines what features correlate to a particular class. Before the last convolutional layer reaches the fully connected layer, the output must be “flattened” from a 3D matrix into a vector.  
 
Pooling
The project particularly focuses on the pooling layer. There are currently two common types of pooling layers: AvgPool and MaxPool. AvgPool takes the average of each area the kernel covers and MaxPool takes the maximum value of each area the kernel covers. Once these values are taken, they form a small matrix, thus downsizing the data. Although there are two known methods, the NNSS has developed another method of pooling that is still being tested and researched: variable stride. Typically, the two pooling methods have the same stride value throughout the matrix; variable stride, however, applies kernels that have different strides in each of its regions with horizontal and vertical value restrictions. Meaning, each row or column can have different values compared to other rows or columns as long as the value remains the same throughout its respective row or column.
 

 

DATA 
The data that will be used are images from eyes affected by diabetic retinopathy. Diabetic retinopathy is a diabetic complication in the eye. It occurs in five stages and has a corresponding number (0-4). Zero having no complications and 4 having the worst. The dataset itself is publicly available and contains images from stage 0 to 4. The dataset that was given by the NNSS is a sample set of pre-processed 600x600 pixel images from the larger dataset. This set contains images only from stage 1 and stage 4. It includes 100 photos for each stage, making it 200 photos in total. 
 
                  

STRATEGY  
To begin this project, we need to build a working neural network that can feed the diabetic retinopathy data into it. We will initially start with MaxPool as our pooling method. Once we get it working,  MaxPool will be replaced with variable stride. Once variable stride is fully implemented into the network, we can begin to compare it with MaxPool and AvgPool. Comparing standards will be based on accuracy and learning rate. Every element of the network will stay the same except for the pooling methods. 
We plan to do our evaluation using the f-score and confusion matrix. F-scores are used to evaluate binary classification systems. It uses the model’s precision and recall producing a mean. Precision and recall are scored by the following: 
 
F-scores can be adjusted to prioritize precision or recall. A confusion matrix is a table that contains the same elements that f-scores measure: true positive, false positive, true negative, and false negative. It visualizes the f-score data. 
      
INITIAL CHALLENGES
Our main challenges stemmed from the variable stride algorithm. Variable stride was coded in python within a class function using Keras. There are three ways to write neural networks using Keras: subclassing, functional API, and sequential API. 

STRIDING CHOICES
 There are many different anomalies that affect an eye with diabetic retinopathy. We wanted our striding to focus on common regions where these anomalies can be found in the retina. Microaneurysms are one of the earlier anomalies that are found in the retina, which occur along the vascular arcades. Neovascularization can commonly occur in the optic disc region. Diabetic macular edema (DME) occurs in the later stages of diabetic retinopathy. Eyes with DME have thickened maculas and have hard exudates We tested three different strides on the eye data. One with focus on the optic disk, one with focus on the macula, and a custom one that focuses on the vascular arcades and optic disk.

	Our results show that there is a small difference between variable stride and the other pooling methods. As mentioned previously, many different anomalies occur in an diabetic retinopathy eye. These anomalies have common tendencies to be in a certain area; however, they can still occur in other areas of the eye even if it is not common. Due to the various locations that these anomalies occur, variable stride could not work as efficiently on this data set. The purpose of variable stride is to focus on important regions of the image, if most of the image contains important features, then variable stride would not work efficiently. 
