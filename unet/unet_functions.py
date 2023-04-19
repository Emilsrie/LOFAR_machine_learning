from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def gray_to_rgb(data):
  new_data = []
  for d in data:
    new_d = color.gray2rgb(d)
    new_data.append(new_d)

  return np.array(new_data)

def img_to_rgb(image):
    return image.convert('RGB')

def prepare_vsrc_image(image):
    image = img_to_rgb(image)
    image = np.asarray(image)
    image = image[:, :-9] # remove white pixels
    image = image[:512, :512]
    print(image.shape)

    return np.asarray(image)

def get_train_test_splits(subset_size, random_state):
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

    return (X_train, X_test, y_train, y_test)


def SaveVisualizedResults(train_test_data, model, index, random_state, unet_version, showplot=False, savefig=False):
    X_train, X_test, y_train, y_test = train_test_data[0], train_test_data[1], train_test_data[2], train_test_data[3],
    img = X_test[index]
    img = img[np.newaxis, ...]
    #print(img.shape)

    pred_y = model.predict(img)
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
        save_path = f'./saved_figs/unet_{unet_version}/'
        isExist = os.path.exists(save_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_path)
        fig.savefig(save_path + f'/r_state{random_state}_idx{index}.png')

    if showplot is True:
        plt.show()