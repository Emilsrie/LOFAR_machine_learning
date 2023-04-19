from tensorflow import keras
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
import tensorflow as tf

# Import time module
import time
# record start time
start = time.time()


def make_imgs_rgb(data):
  new_data = []
  for idx, d in enumerate(data):
    #d = cv2.normalize(d, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #d = (d * 255).astype(np.uint8)
    new_d = color.gray2rgb(d)
    new_data.append(new_d)

  return np.array(new_data)


unet = keras.models.load_model('./saved models/saved_unet_V2/')

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

x_input, y_input = 512, 512

random_state=100
# Use scikit-learn's function to split the dataset
# Here, I have used 20% data as test/valid set
X_train, X_test, y_train, y_test = train_test_split(image_data, mask_data, test_size=0.2, random_state=random_state)
X_train = make_imgs_rgb(X_train)
X_test = make_imgs_rgb(X_test)

#default learning rate used value 0.001
results = unet.fit(X_train,
                   y_train,
                   batch_size=5,
                   epochs=1,
                   validation_data=(X_test, y_test))

fig, axis = plt.subplots(1, 2, figsize=(20, 5))
axis[0].plot(results.history["loss"], color='r', label='train loss')
axis[0].plot(results.history["val_loss"], color='b', label='validation loss')
axis[0].set_title('Loss Comparison')
axis[0].legend()
axis[1].plot(results.history["accuracy"], color='r', label='train accuracy')
axis[1].plot(results.history["val_accuracy"], color='b', label='validation accuracy')
axis[1].set_title('Accuracy Comparison')
axis[1].legend()

def VisualizeResults(index, savefig=False):
    img = X_test[index]
    img = img[np.newaxis, ...]
    print(img.shape)

    pred_y = unet.predict(img)
    pred_mask = tf.argmax(pred_y[0], axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]

    pred_mask_np = pred_mask.numpy()
    pred_mask_np[pred_mask_np > 0] = 1

    print(pred_mask_np.shape)
    unique, counts = np.unique(pred_mask_np, return_counts=True)
    print(dict(zip(unique, counts)))

    fig = plt.figure(figsize=(15, 15))
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

    plt.tight_layout()
    plt.show()

    fig.tight_layout()
    if savefig == True:
        fig.savefig(f'./saved_figs/r_state{random_state}_idx{index}.png')

print(unet.evaluate(X_test, y_test))

# save model
unet.save('./saved models/saved_unet_V3', overwrite=True)


index = 2
VisualizeResults(index, savefig=True)

"""
Really good picture with random_state=100 and index 2
"""

# record end time
end = time.time()

# print the difference between start
# and end time in milli. secs
working_time = (end - start) * 10 ** 3
print("The time of execution of above program is :",
     working_time, "ms")

with open('time.txt', 'w') as f:
    f.write(str(working_time))
