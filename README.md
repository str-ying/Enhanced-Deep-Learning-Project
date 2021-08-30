# Enhanced-Deep-Learning-Project

 

INTRODUCTION 
Nevada National Security Site (NNSS) has partnered with Embry Riddle Aeronautical University (ERAU) to collaborate in a research project regarding a novel pooling method that the NNSS has developed: variable stride. The NNSS is a high security support site to national laboratories. The NNSS conducts research and experiments with nuclear weapons, national security, and various environmental programs. They have a multi-faceted mission, a part of it being the Stewardship Science Experimentation (SSE). The SSE is centered around the National Nuclear Security Administration (NNSA) which does various non nuclear experiments. The NNSS hosts one of the SSE’s primary labs, the U1a underground complex which experiments for the nation’s subcritical experimentation program. A form of data collecting at this lab is through x-rays machines which produce images during experiments. The lab uses image processing methods to analyze the data and one method of image processing are convolutional neural networks. The NNSS has developed a new pooling method called variable stride. This project is supported by the National Science Foundation (NSF) through REU award number DMS - 2050754. 
 
PROJECT SCOPE
 The main goal of this project is to examine variable stride. Since it has been developed recently, it has not been used or tested compared to the other pooling methods. To examine this method, a small data set of diabetic retinopathy eye images were given to use to implement into the convolutional neural network (CNN). Variable stride will be evaluated using metrics such as the ROC curve. Along with the examination, variable stride will be compared to AvgPool and MaxPool.
 
BACKGROUND
Neural Networks 
CNNs are deep neural networks that can identify specific aspects of an image when trained. Neural networks are inspired by the human neural system. Each neuron is interconnected with each other to form a network. The neurons or nodes each have a weight which determines the importance of the neuron. The neuron is multiplied by its weight and summed up and passed into an activation function that calculates the output of the neuron. This process repeats for each layer and is called forward propagation. The network also does back propagation which takes the error rate of the output of the previous epoch and finetunes the weights using partial derivatives. Back propagation is essential for neural network training due to its fine tuning of the weights. The distinguishing factor of a CNN is its convolutional layer and pooling layer.

Convolutional Neural Network Layers
Within a CNN there are multiple layers that work to help identify the image: convolutional layer, pooling layer, and fully connected layer. The convolutional layer applies filters or weights that have values to help the machine detect features. The filters have stride and size that lets it know how many “steps” to take within the picture. The filters themselves are matrices that yield one number when applied to the image. Each convolutional layer is followed by an activation function. The pooling layer downsizes data to reduce the size of the matrix. The fully connected layer looks at the last convolutional layer and determines what features correlate to a particular class. Before the last convolutional layer reaches the fully connected layer, the output must be “flattened” from a 3D matrix into a vector.  
 
Pooling
The project particularly focuses on the pooling layer. There are currently two common types of pooling layers: AvgPool and MaxPool. AvgPool takes the average of each area the kernel covers and MaxPool takes the maximum value of each area the kernel covers. Once these values are taken, they form a small matrix, thus downsizing the data. Although there are two known methods, the NNSS has developed another method of pooling that is still being tested and researched: variable stride. Typically, the two pooling methods have the same stride value throughout the matrix; variable stride, however, applies kernels that have different strides in each of its regions with horizontal and vertical value restrictions. Meaning, each row or column can have different values compared to other rows or columns as long as the value remains the same throughout its respective row or column.
 
DATA 
The data that will be used are images from eyes affected by diabetic retinopathy. Diabetic retinopathy is a diabetic complication in the eye. It occurs in five stages and has a corresponding number (0-4). Zero having no complications and 4 having the worst. The dataset itself is publicly available and contains images from stage 0 to 4. The dataset that was given by the NNSS is a sample set of pre-processed 600x600 pixel images from the larger dataset. This set contains images only from stage 1 and stage 4. It includes 100 photos for each stage, making it 200 photos in total. 
 
STRIDING CHOICES
 The dataset that was used for this experiment was images of eyes that are affected by class 1 and class 4 diabetic retinopathy. There are many different anomalies that affect an eye with diabetic retinopathy. Since variable stride focuses on important features of an image, we wanted our striding to focus on common regions where these anomalies can be found in the retina. We tested three different strides on the eye data. One with focus on the right which is the optic disk. Neovascularization can commonly occur in the optic disc region along with the chances of other anomalies. Our second stride focuses on the center which is the macula. Diabetic macular edema (DME) occurs in the macula during the later stages of diabetic retinopathy. Eyes with DME have thickened maculas and have hard exudates.  Finally, we have a custom stride that focuses on the vascular arcades and optic disk. Microaneurysms are one of the earlier anomalies that are found in the retina, which occur along the vascular arcades. The custom stride would allow for easier detection of microaneurysms while detecting anomalies in the optic disc. 

Custom – Vertical Stride: [2, 3, 1, 3, 2], Vertical Fraction: [1/5, 1/5, 1/5, 1/5, 1/5], Horizontal Stride: [3, 2], Horizontal Fraction: [1/5, 4/5]
Right – Vertical Stride: [3, 1, 3], Vertical Fraction: [1/3, 1/3, 1/3], Horizontal Stride: [2, 3], Horizontal Fraction: [2/3, 1/3]
Center – Vertical Stride: [4, 1, 4], Vertical Fraction: [1/3, 1/3, 1/3], Horizontal Stride: [4, 1, 4], Horizontal Fraction: [1/3, 1/3, 1/3]

INITIAL CHALLENGES
Our main challenges stemmed from the variable stride algorithm. There are three ways to write neural networks using Keras: subclassing, functional API, and sequential API. The NNSS embedded variable stride in a class function using sequential API. Since it was in a class function, the line of code that was given to implement variable stride into the code included the keyword self. Our initial implementation of the algorithm was unsuccessful due to the embedded sequential API model in a class function; however, we were able to rewrite the algorithm as a pre-processing method instead of implementing it as a layer in the model. Both the pre-processing method and the layer implementation work the same. 

EXPERIMENT AND RESULTS 
Experiment
The experiment was created based on the results from a preliminary analysis of different network structures and eye anomaly locations. The networks with the highest accuracy when trained on the data were chosen: 
 64 Node Network, 5 Convolutional Layers, 4 Dense Layers, and .05 Dropout Rate .8000
 32 Node Network, 5 Convolutional Layers, 4 Dense Layers, and .00 Dropout Rate .7250
16 Node Network, 5 Convolutional Layers, 1 Dense Layers, and .05 Dropout Rate .7500
During the experiment, we tested 5 different pooling methods on each of the three networks. The pooling methods included the three variable stride schemes stated previously, MaxPool, and AvgPool. We recorded loss/accuracy, validation loss/accuracy, ROC curve, AUC, precision, and recall for each of the runs. 
Results
	To analyze our results, we took each metric and compared the networks against each other using a t-test. One metric is not enough to make a conclusion about a pooling method, so we look at the overall performance of the pooling methods along 4 different metrics: validation accuracy, validation loss, AUC, and epoch. We demonstrated our results in a comparison chart for each metric.
	
CONCLUSION
Although most of the comparisons had no significance, the comparisons that did have significant results showed that MaxPool and AvgPool performed better than Variable Stride. There can be two factors that can contribute to Variable Stride’s underperformance. We tested our networks using a dataset of 200 images. Performance results might differ if we tested on a larger dataset since, the more training data you have, the better network performance. Another factor is the dataset itself. The complications in diabetic retinopathy are not focused on one area of the eye, they’re scattered around the eye. Since Variable Stride focuses on important areas in an image, the diabetic retinopathy dataset could not be an ideal dataset for this pooling method due to the nature of where these anomalies are located. 


[cnnPosterupdated.pdf](https://github.com/str-ying/Enhanced-Deep-Learning-Project/files/7079724/cnnPosterupdated.pdf)
