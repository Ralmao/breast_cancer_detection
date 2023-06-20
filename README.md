breast_cancer_detection
Using Neural Networks for a breast cancer detection
Libraries:
import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import cv2
import tensorflow as tfm
import tensorflow as tf
from google.colab import drive

dataset from : https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

Explaning a segmentation with convolutional neural networks:

Semantic segmentation is the task of assigning a class label to each pixel in an image. This is different from image classification, where the entire image is assigned a single class label. Convolutional neural networks (CNNs) are a powerful tool for semantic segmentation, as they can learn to identify patterns in images at different scales.

CNNs work by sliding a filter over an image, and computing the dot product between the filter and the image pixels. This process is repeated multiple times, with different filters each time. The output of the CNN is a feature map, which represents the different features that have been detected in the image.

For semantic segmentation, the feature maps are used to classify each pixel in the image. This is done by using a fully connected layer to map the feature maps to a set of class labels. The output of the fully connected layer is a probability distribution over the class labels, for each pixel in the image.

CNNs have been shown to be very effective for semantic segmentation. They have been used to segment a wide variety of images, including natural images, medical images, and satellite images.

Here are some of the benefits of using CNNs for semantic segmentation:

CNNs are able to learn to identify patterns in images at different scales. This is important for semantic segmentation, as different objects in an image can have different sizes. CNNs are able to learn to ignore irrelevant information in images. This is important for semantic segmentation, as there is often a lot of noise in images. CNNs are able to learn to segment images very accurately. This is important for many applications, such as self-driving cars and medical image analysis. Here are some of the challenges of using CNNs for semantic segmentation:

CNNs require a large amount of training data. This can be a challenge, as it can be difficult to collect large datasets of images that have been manually segmented. CNNs can be computationally expensive to train. This can be a challenge, as it can require a lot of computing resources. CNNs can be sensitive to the choice of hyperparameters. This means that it can be difficult to find the optimal hyperparameters for a particular dataset. Overall, CNNs are a powerful tool for semantic segmentation. They have been shown to be very effective for a wide variety of tasks. However, there are some challenges associated with using CNNs, such as the need for large datasets and the sensitivity to hyperparameters.
