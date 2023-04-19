import pickle
import matplotlib.pyplot as plt
datapath = 'C:/Users/emilsrie/VirtualBox_VMs/shared_folder/LOFAR_Full_RFI_dataset.pkl'

# import pandas as pd
# original_df = pd.read_pickle(dataset_path)
# #print(original_df.head())

subset_num = 1000

with open(datapath, 'rb') as f:
    data = pickle.load(f)
    print(type(data))
    print(data[0][0].shape)
    data = data[0][:subset_num]

with open(f'LOFAR_subset_{subset_num}.pkl', 'wb') as f:
    pickle.dump(data, f)

image = data[0]
fig = plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.imshow(image, interpolation='nearest')
plt.tight_layout()
plt.show()

