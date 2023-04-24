import os
import numpy as np
from skimage import io
from skimage import color
import cv2
from PIL import Image
import pickle

image_directory = 'C:/Users/emilsrie/Desktop/lofar_observations/results/'

image_save_directory = 'C:/Users/emilsrie/Desktop/lofar_observations/vsrc_lofar_for_training/images/'
#cleaned_image_save_directory = 'C:/Users/emilsrie/Desktop/lofar_observations/vsrc_lofar_for_training/cleaned_images/'
rfi_mask_save_directory = 'C:/Users/emilsrie/Desktop/lofar_observations/vsrc_lofar_for_training/mask_images/'

image_paths = []
#cleaned_image_paths = []
mask_paths = []

def get_dirs(rootdir):
    for file in os.listdir(rootdir):
        data_file = os.path.join(rootdir, file)

        if os.path.isdir(data_file):
            get_dirs(data_file)

        elif os.path.isfile(data_file):
            if '_ax1_figure' in data_file:
                image_paths.append(data_file)
            # if 'sumax1_figure' in data_file:
            #     cleaned_image_paths.append(data_file)
            if 'just_rfiax1_figure' in data_file:
                mask_paths.append(data_file)


def get_images(image_paths):
    images = []
    for im_path in image_paths:
        #img = Image.open(im_path) #RGBA
        img = io.imread(im_path)
        images.append(img)
    return images

def generate_images(images, save_directory, pkl_file_name):
    generated_images = []

    for idx, im in enumerate(images):
        #im = color.rgba2rgb(im)
        #im = color.rgb2gray(im)
        im1 = im[:992, :992, :]
        im2 = im[:992, 800:1792, :]
        # flip image
        im3 = cv2.flip(im1, 1)
        im4 = cv2.flip(im2, 1)

        im1 = Image.fromarray(im1)
        im2 = Image.fromarray(im2)
        im3 = Image.fromarray(im3)
        im4 = Image.fromarray(im4)

        im1.save(save_directory + str(idx+1) + '_1.png')
        im2.save(save_directory + str(idx+1) + '_2.png')
        im3.save(save_directory + str(idx+1) + '_3.png')
        im4.save(save_directory + str(idx+1) + '_4.png')

        im1 = np.asarray(im1)
        im2 = np.asarray(im2)
        im3 = np.asarray(im3)
        im4 = np.asarray(im4)

        generated_images.append(im1)
        generated_images.append(im2)
        generated_images.append(im3)
        generated_images.append(im4)

    generated_images = np.asarray(generated_images)
    with open(save_directory + f'{pkl_file_name}.pkl', 'wb') as f:
        pickle.dump(generated_images, f)


get_dirs(image_directory)

#images = get_images(image_paths)
#cleaned_images = get_images(cleaned_image_paths)
rfi_mask_images = get_images(mask_paths)

#generate_images(images, image_save_directory, 'images')
#generate_images(cleaned_images, cleaned_image_save_directory, 'cleaned_images')
generate_images(rfi_mask_images, rfi_mask_save_directory, 'rfi_mask_images')
