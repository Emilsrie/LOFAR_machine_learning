import matplotlib.pyplot as plt
import unet_functions as u_f
from PIL import Image
from tensorflow import keras

path = '../LOFAR/VSRC_data/'
file = 'bst_SUN_figure1.png'

img = Image.open(path + file)
img = u_f.prepare_vsrc_image(img)

plt.rcParams['figure.figsize'] = [10, 10]
plt.subplot(111)
plt.imshow(img)
plt.show()

