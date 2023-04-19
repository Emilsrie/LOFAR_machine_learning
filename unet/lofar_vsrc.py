import matplotlib.pyplot as plt
import unet_functions as u_f
from PIL import Image
from tensorflow import keras
#import aoflagger
import numpy as np
from skimage import color
import tensorflow as tf


path = '../LOFAR/VSRC_data/'
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


#data = flag_img(data)
#np.save('vsrc_masks', data)

masks = np.load('C:/Users/emilsrie/Documents/GitHub/LOFAR_machine_learning/unet/vsrc_masks.npy')

random_state = 100
subset_size = 1000
unet_version = 'V1'
unet = keras.models.load_model(f'saved models/saved_unet_{unet_version}/')

X_test, y_test = data[0], masks[0]

print(unet.evaluate(X_test, y_test))

img = X_test
img = img[np.newaxis, ...]
# print(img.shape)

pred_y = unet.predict(img)
pred_mask = tf.argmax(pred_y[0], axis=-1)
pred_mask = pred_mask[..., tf.newaxis]

pred_mask_np = pred_mask.numpy()
pred_mask_np[pred_mask_np > 0] = 1

# print(pred_mask_np.shape)
unique, counts = np.unique(pred_mask_np, return_counts=True)
print(dict(zip(unique, counts)))

fig = plt.figure(figsize=(15, 15))
fig.tight_layout()
plt.subplot(131)
plt.imshow(color.rgb2gray(X_test))
plt.title('Sākotnējais attēls')
plt.xlabel('Laiks [s]')
plt.ylabel('Frekvence')

plt.subplot(132)
plt.imshow(y_test)
plt.title('AOFlagger maska')
plt.xlabel('Laiks [s]')
plt.ylabel('Frekvence')

plt.subplot(133)
plt.imshow(pred_mask)
plt.title('Prognozētie RFI pikseļi')
plt.xlabel('Laiks [s]')
plt.ylabel('Frekvence')


fig.savefig(f'bst_SUN_figure1_masked.png')

plt.show()



