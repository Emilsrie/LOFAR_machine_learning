from tensorflow import keras
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
import tensorflow as tf
import os

random_state = 100
subset_size = 1000
unet_version = 'V1'


def make_imgs_rgb(data):
  new_data = []
  for d in data:
    new_d = color.gray2rgb(d)
    new_data.append(new_d)

  return np.array(new_data)


path = '../LOFAR/LOFAR subset 100/'
images_path = path + 'LOFAR_subset_100.pkl'
masks_path = path + 'LOFAR_subset_100_masks.pkl'

with open(images_path, 'rb') as f:
    image_data = pickle.load(f)

with open(masks_path, 'rb') as f:
    mask_data = pickle.load(f)

# Preprocessing
# Prepare data
image_data = np.squeeze(image_data, axis=-1)


random_state=100
# Use scikit-learn's function to split the dataset
# Here, I have used 20% data as test/valid set
X_train, X_test, y_train, y_test = train_test_split(image_data, mask_data, test_size=0.2, random_state=random_state)
X_train = make_imgs_rgb(X_train)
X_test = make_imgs_rgb(X_test)


unet = keras.models.load_model(f'saved models/saved_unet_{unet_version}/')
print(unet.evaluate(X_test, y_test))

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

    if showplot is True:
        plt.show()

    if savefig is True:
        # python program to check if a directory exists
        # Check whether the specified path exists or not
        save_path = f'./unet/saved_figs/unet_{unet_version}/'
        isExist = os.path.exists(save_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_path)
        fig.savefig(save_path + f'/r_state{random_state}_idx{index}.png')


index = 19
VisualizeResults(index, True, False)
"""
Really good picture with unet_V1, random_state=100 and index 2
Really good picture with unet_V1, random_state=100 and index 19
"""
