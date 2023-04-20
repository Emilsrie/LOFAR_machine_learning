import matplotlib.pyplot as plt
import unet_functions as u_f
from PIL import Image
from tensorflow import keras
#import aoflagger
import numpy as np
from skimage import color
import tensorflow as tf

def img_to_rgb(image):
    return image.convert('RGB')

def prepare_vsrc_image(image):
    image = img_to_rgb(image)
    image = np.asarray(image)
    image = image[:, :-9] # remove white pixels
    image = image[:512, :512]
    print(image.shape)

    return np.asarray(image)


#path = './LOFAR/VSRC_data/'
path = 'C:/Users/emilsrie/Documents/GitHub/LOFAR_machine_learning/LOFAR/VSRC_data/'
file = 'bst_SUN_figure1.png'

img = Image.open(path + file)
img = u_f.prepare_vsrc_image(img)
print(img.shape)

img = color.rgb2gray(img)
data = []
data.append(img)

# def flag_img(data, do_plot=False):
#     masks = []
#     for idx, image in enumerate(data):
#         nch = image.shape[0]
#         ntimes = image.shape[1]

#         flagger = aoflagger.AOFlagger()
#         # path = flagger.find_strategy_file(aoflagger.TelescopeId.LOFAR)
#         strategy = flagger.load_strategy_file('/home/emilsrie/MyProjects/LOFAR_machine_learning/aoflag/strategies/lofar_strat.lua')

#         data = flagger.make_image_set(ntimes, nch, 1)
#         data.set_image_buffer(0, image)  # Real values

#         flags = strategy.run(data)
#         flag_mask = flags.get_buffer()

#         masks.append(flag_mask)

#     return np.array(masks)




# masks = []
# masks.append(np.expand_dims(flag_img(data), axis=-1))
# masks = np.asarray(masks)
# np.save('vsrc_masks', data)

img = u_f.gray_to_rgb(img)
print(img.shape)
data = []
data.append(img)

masks = np.load('C:/Users/emilsrie/Documents/GitHub/LOFAR_machine_learning/vsrc_masks.npy')

random_state = 100
subset_size = 1000
unet_version = 'V1'
#unet = keras.models.load_model(f'unet/saved models/saved_unet_{unet_version}/')
unet = keras.models.load_model(f'C:/Users/emilsrie/Documents/GitHub/LOFAR_machine_learning/unet/saved models/saved_unet_{unet_version}/')

X_train, X_test, y_train, y_test = data, data, masks, masks
index = 0
u_f.SaveVisualizedResults((X_train, X_test, y_train, y_test), unet, index, random_state, unet_version, True, True)






