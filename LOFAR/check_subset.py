import pickle
import matplotlib.pyplot as plt
datapath = '../LOFAR/LOFAR subset 1000/LOFAR_subset_1000.pkl'

# import pandas as pd
# original_df = pd.read_pickle(dataset_path)
# #print(original_df.head())


with open(datapath, 'rb') as f:
    data = pickle.load(f)
    print(type(data))
    print(len(data))
    print(data[0].shape)


image = data[400]
fig = plt.figure(figsize=(10, 10))
plt.imshow(image, interpolation='nearest')
plt.tight_layout()
plt.show()
