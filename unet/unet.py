# -*- coding: utf-8 -*-

"""
###Unet V2 that is using LOFAR image set (100 or 1000 images out of 74k) with the size 512x512
dataset based on: https://github.com/mesarcik/RFI-NLN/tree/d3a7b1d662422518c1d343d4cf5ac81d40e45723
"""


import numpy as np
import matplotlib.pyplot as plt
from skimage import color
# for bulding and running deep learning model
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Conv2DTranspose
from keras.layers import concatenate
from sklearn.model_selection import train_test_split
import os
import pickle

# Don't display errors
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Check GPU compatability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))


def gray_to_rgb(data):
  new_data = []
  for d in data:
    new_d = color.gray2rgb(d)
    new_data.append(new_d)

  return np.array(new_data)

random_state = 100
subset_size = 100

path = f'../LOFAR/LOFAR subset {subset_size}/'
images_path = path + f'LOFAR_subset_{subset_size}.pkl'
masks_path = path + f'LOFAR_subset_{subset_size}_masks.pkl'

with open(images_path, 'rb') as f:
    image_data = pickle.load(f)

with open(masks_path, 'rb') as f:
    mask_data = pickle.load(f)
    image_data = np.squeeze(image_data, axis=-1)

X_train, X_test, y_train, y_test = train_test_split(image_data, mask_data, test_size=0.2, random_state=random_state)
X_train = gray_to_rgb(X_train)
X_test = gray_to_rgb(X_test)

x_input, y_input = 512, 512

"""Smthn smthn false flagging rate here"""

fig = plt.figure(figsize=(20, 25))
plt.subplot(121)
plt.imshow(color.rgb2gray(X_train[0]))

plt.subplot(122)
plt.imshow(y_train[0])

plt.show()
"""# 2. Create the model

# 2.1 - U-Net Encoder Block
"""

def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    """
    This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning. 
    Dropout can be added for regularization to prevent overfitting. 
    The block returns the activation values for next layer along with a skip connection which will be used in the decoder
    """
    # Add 2 Conv Layers with relu activation and HeNormal initialization using TensorFlow 
    # Proper initialization prevents from the problem of exploding and vanishing gradients 
    # 'Same' padding will pad the input to conv layer such that the output has the same height and width (hence, is not reduced in size) 
    conv = Conv2D(n_filters, 
                  3,   # Kernel size   
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)
    conv = Conv2D(n_filters, 
                  3,   # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)
    
    # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
    conv = BatchNormalization()(conv, training=False)

    # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
    if dropout_prob > 0:     
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    # Pooling reduces the size of the image while keeping the number of channels same
    # Pooling has been kept as optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use)
    # Below, Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse across input image
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv

    # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during transpose convolutions      
    skip_connection = conv
    
    return next_layer, skip_connection

"""# 2.2 - U-Net Decoder Block"""

def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
    """
    Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output
    """
    # Start with a transpose convolution layer to first increase the size of the image
    up = Conv2DTranspose(
                 n_filters,
                 (3, 3),    # Kernel size
                 strides=(2, 2),
                 padding='same')(prev_layer_input)

    # Merge the skip connection from previous block to prevent information loss
    merge = concatenate([up, skip_layer_input], axis=3)
    
    # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
    # The parameters for the function are similar to encoder
    conv = Conv2D(n_filters, 
                 3,     # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(merge)
    conv = Conv2D(n_filters,
                 3,   # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(conv)
    return conv

"""# 2.3 - Compile U-Net Blocks"""

def UNetCompiled(input_size=(x_input, y_input, 3), n_filters=32, n_classes=2):
    """
    Combine both encoder and decoder blocks according to the U-Net research paper
    Return the model as output 
    """
    # Input size represent the size of 1 image (the size used for pre-processing) 
    inputs = Input(input_size)

    # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
    # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of the image 
    cblock1 = EncoderMiniBlock(inputs, n_filters, dropout_prob=0, max_pooling=True)
    cblock2 = EncoderMiniBlock(cblock1[0], n_filters*2, dropout_prob=0, max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters*4, dropout_prob=0, max_pooling=True)
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters*8, dropout_prob=0.3, max_pooling=True)
    cblock5 = EncoderMiniBlock(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False) 

    # Decoder includes multiple mini blocks with decreasing number of filters
    # Observe the skip connections from the encoder are given as input to the decoder
    # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used
    ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1],  n_filters * 8)
    ublock7 = DecoderMiniBlock(ublock6, cblock3[1],  n_filters * 4)
    ublock8 = DecoderMiniBlock(ublock7, cblock2[1],  n_filters * 2)
    ublock9 = DecoderMiniBlock(ublock8, cblock1[1],  n_filters)

    # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
    # Followed by a 1x1 Conv layer to get the image to the desired size. 
    # Observe the number of channels will be equal to number of output classes
    conv9 = Conv2D(n_filters,
                3,
                activation='relu',
                padding='same',
                kernel_initializer='he_normal')(ublock9)

    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model

"""#3.2 - Build U-Net Architecture"""

# Call the helper function for defining the layers for the model, given the input image size
unet = UNetCompiled(input_size=(x_input, y_input, 3), n_filters=32, n_classes=3)

# Check the summary to better interpret how the output dimensions change in each layer
unet.summary()

"""# 3.2 - Compile and Run Model"""

# There are multiple optimizers, loss functions and metrics that can be used to compile multi-class segmentation models
# Ideally, try different options to get the best accuracy
unet.compile(optimizer=tf.keras.optimizers.Adam(), 
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])

# Run the model in a mini-batch fashion and compute the progress for each epoch
results = unet.fit(X_train, 
                   y_train, 
                   batch_size=16,
                   epochs=4,
                   validation_data=(X_test, y_test))

"""# 4 - Evaluate Model Results

## 4.1 - Bias Variance Check
"""

# High Bias is a characteristic of an underfitted model and we would observe low accuracies for both train and validation set
# High Variance is a characterisitic of an overfitted model and we would observe high accuracy for train set and low for validation set
# To check for bias and variance plit the graphs for accuracy 
# I have plotted for loss too, this helps in confirming if the loss is decreasing with each iteration - hence, the model is optimizing fine
fig, axis = plt.subplots(1, 2, figsize=(20, 5))
axis[0].plot(results.history["loss"], color='r', label='train loss')
axis[0].plot(results.history["val_loss"], color='b', label='validation loss')
axis[0].set_title('Loss Comparison')
axis[0].legend()
axis[1].plot(results.history["accuracy"], color='r', label='train accuracy')
axis[1].plot(results.history["val_accuracy"], color='b', label='validation accuracy')
axis[1].set_title('Accuracy Comparison')
axis[1].legend()

# RESULTS
# The train loss is consistently decreasing showing that Adam is able to optimize the model and find the minima


"""## 4.2 - View Predicted Segmentations#"""
print(unet.evaluate(X_test, y_test))

# save model
unet_version = 'V3_100'
unet.save(f'../unet/saved models/saved_unet_{unet_version}', overwrite=True)

def VisualizeResults(index, showplot=False, savefig=False):
    img = X_test[index]
    img = img[np.newaxis, ...]
    #print(img.shape)

    pred_y = unet.predict(img)
    pred_mask = tf.argmax(pred_y[0], axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]

    pred_mask_np = pred_mask.numpy()
    pred_mask_np[pred_mask_np > 0] = 1

    #print(pred_mask_np.shape)
    unique, counts = np.unique(pred_mask_np, return_counts=True)
    print(dict(zip(unique, counts)))

    fig = plt.figure(figsize=(15, 15))
    fig.tight_layout()
    plt.subplot(131)
    plt.imshow(color.rgb2gray(X_test[index]))
    plt.title('Sākotnējais attēls')
    plt.xlabel('Laiks [s]')
    plt.ylabel('Frekvence')

    plt.subplot(132)
    plt.imshow(y_test[index])
    plt.title('AOFlagger maska')
    plt.xlabel('Laiks [s]')
    plt.ylabel('Frekvence')

    plt.subplot(133)
    plt.imshow(pred_mask)
    plt.title('Prognozētie RFI pikseļi')
    plt.xlabel('Laiks [s]')
    plt.ylabel('Frekvence')


    if savefig is True:
        # python program to check if a directory exists
        # Check whether the specified path exists or not
        save_path = f'./unet/saved_figs/unet_{unet_version}/'
        isExist = os.path.exists(save_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_path)
        fig.savefig(save_path + f'/r_state{random_state}_idx{index}.png')

    if showplot is True:
        plt.show()

# Add any index to contrast the predicted mask with actual mask
index = 3
VisualizeResults(index, showplot=False, savefig=False)




