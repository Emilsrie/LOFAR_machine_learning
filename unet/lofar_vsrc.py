import matplotlib.pyplot as plt
import unet_functions as u_f
from PIL import Image
from tensorflow import keras
import aoflagger
import numpy as np
from skimage import color
import tensorflow as tf


path = './LOFAR/VSRC_data/'
file = 'bst_SUN_figure1.png'

img = Image.open(path + file)
img = u_f.prepare_vsrc_image(img)
print(img.shape)

img = color.rgb2gray(img)
data = []
data.append(img)

def flag_img(data, do_plot=False):
    masks = []
    for idx, image in enumerate(data):
        nch = image.shape[0]
        ntimes = image.shape[1]

        flagger = aoflagger.AOFlagger()
        # path = flagger.find_strategy_file(aoflagger.TelescopeId.LOFAR)
        strategy = flagger.load_strategy_file('/home/emilsrie/MyProjects/LOFAR_machine_learning/aoflag/strategies/lofar_strat.lua')

        data = flagger.make_image_set(ntimes, nch, 1)
        data.set_image_buffer(0, image)  # Real values

        flags = strategy.run(data)
        flag_mask = flags.get_buffer()

        masks.append(flag_mask)

    return np.array(masks)


masks = flag_img(data)
print(masks[0].shape)
np.save('vsrc_masks', data)

#masks = np.load('../LOFAR_machine_learning/unet/vsrc_masks.npy')

random_state = 100
subset_size = 1000
unet_version = 'V1'
unet = keras.models.load_model(f'unet/saved models/saved_unet_{unet_version}/')

X_train, X_test, y_train, y_test = data[0], data[0], masks[0], masks[0]
index = 0
u_f.SaveVisualizedResults((X_train, X_test, y_train, y_test), unet, index, random_state, unet_version, True, True)






