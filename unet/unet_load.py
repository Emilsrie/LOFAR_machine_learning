from tensorflow import keras
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image

random_state = 100
subset_size = 100
unet_version = 'V4_test'


def normalize(im):
    return ((im - im.min()) * (1 / (im.max() - im.min()) * 255)).astype('uint8')


def make_imgs_rgb(data):
  new_data = []
  for im in data:
    im = color.gray2rgb(im)
    new_data.append(im)

  return np.array(new_data)


path = f'../LOFAR/LOFAR subset {subset_size}/'
images_path = path + f'LOFAR_subset_{subset_size}.pkl'
masks_path = path + f'LOFAR_subset_{subset_size}_masks.pkl'

with open(images_path, 'rb') as f:
    image_data = pickle.load(f)
    #image_data = np.squeeze(image_data, axis=-1)

with open(masks_path, 'rb') as f:
    mask_data = pickle.load(f)
    mask_data = np.expand_dims(mask_data, axis=-1)

# Preprocessing
# Prepare data





# Use scikit-learn's function to split the dataset
# Here, I have used 20% data as test/valid set
X_train, X_test, y_train, y_test = train_test_split(image_data, mask_data, test_size=0.2, random_state=random_state)
#X_train = make_imgs_rgb(X_train)
#X_test = make_imgs_rgb(X_test)

#image_data = make_imgs_rgb(image_data)

unet = keras.models.load_model(f'saved models/saved_unet_{unet_version}/')
unet.save_weights(f'../unet/saved model weights/saved_unet_{unet_version}', overwrite=True)

print(unet.evaluate(X_test, y_test))


#def VisualizeResults(index, showplot=False, savefig=False):
def VisualizeResults(index, showplot=False, savefig=False):
    #img = X_test[index]
    img = image_data[index]
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
    #plt.imshow(color.rgb2gray(X_test[index]))

    # Increase contrast
    #original_im = color.rgb2gray(image_data[index])
    original_im = image_data[index]
    normalized_im = normalize(original_im)
    #print(normalized_im)
    # print(normalized_im.shape)
    #print(np.min(normalized_im))
    #print(np.max(normalized_im))


    plt.imshow(normalized_im, vmin=0, vmax=15)
    plt.imshow(original_im)
    plt.title('Sākotnējais attēls')
    plt.xlabel('Laiks [s]')
    plt.ylabel('Frekvence')

    plt.subplot(132)
    #plt.imshow(y_test[index])
    plt.imshow(mask_data[index])
    plt.title('AOFlagger maska')
    plt.xlabel('Laiks [s]')
    plt.ylabel('Frekvence')

    plt.subplot(133)
    plt.imshow(pred_mask)
    plt.title('Prognozētie RFI pikseļi')
    plt.xlabel('Laiks [s]')
    plt.ylabel('Frekvence')

    plt.tight_layout()

    if showplot is True:
        plt.show()

    if savefig is True:
        # python program to check if a directory exists
        # Check whether the specified path exists or not
        save_path = f'./saved_figs/unet_{unet_version}/'
        isExist = os.path.exists(save_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_path)
        fig.savefig(save_path + f'/r_state{random_state}_idx{index}.png')

    plt.close()


index = 81
VisualizeResults(index, True, False)

# for index in range(100):
#     print(index)
#     VisualizeResults(index, False, True)

