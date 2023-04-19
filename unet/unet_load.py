from tensorflow import keras
import unet_functions as u_f

random_state = 100
subset_size = 1000
unet_version = 'V1'
unet = keras.models.load_model(f'saved models/saved_unet_{unet_version}/')

train_test_data = u_f.get_train_test_splits(subset_size, random_state)
X_train, X_test, y_train, y_test = train_test_data[0], train_test_data[1], train_test_data[2], train_test_data[3],

print(unet.evaluate(X_test, y_test))
for index in range(200):
    u_f.SaveVisualizedResults(train_test_data, unet, index, random_state, unet_version, showplit=True, savefig=False)

"""
Really good picture with unet_V1, random_state=100 and index 2
Really good picture with unet_V1, random_state=100 and index 19
"""
