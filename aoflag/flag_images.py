import aoflagger
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import color
import pickle
import os

image_directory = '../LOFAR_machine_learning/LOFAR/LOFAR subset 100/'
save_directory = '../LOFAR_machine_learning/LOFAR/LOFAR subset 100/'

with open(image_directory + 'LOFAR_subset_100.pkl', 'rb') as f:
    data = pickle.load(f)


def flag(data, do_plot=False):
    masks = []
    for idx, image in enumerate(data):
        image = np.squeeze(image, axis=-1)

        nch = image.shape[0]
        ntimes = image.shape[1]
        count = 1  # 50    # number of trials in the false-positives test

        flagger = aoflagger.AOFlagger()
        # path = flagger.find_strategy_file(aoflagger.TelescopeId.LOFAR)
        strategy = flagger.load_strategy_file('/home/emilsrie/MyProjects/LOFAR_machine_learning/aoflag/strategies/lofar_strat.lua')

        data = flagger.make_image_set(ntimes, nch, 1)
        data.set_image_buffer(0, image)  # Real values

        flags = strategy.run(data)
        flag_mask = flags.get_buffer()
        rfi_pixels = flag_mask * image

        masks.append(flag_mask)


        # if flip == True:
        #     np.save(save_directory + "/" + filename, flag_mask.T)
        # else:
        #     np.save(save_directory + "/" + filename, flag_mask)

        if do_plot == True:
            fig = plt.figure(figsize=(20, 10))
            plt.subplot(131)
            plt.imshow(image, interpolation='nearest')
            plt.title('LOFAR attēls {}'.format(idx))
            plt.subplot(132)
            plt.imshow(flag_mask, interpolation='nearest')
            plt.title('Segmentētā maska')
            plt.subplot(133)
            plt.imshow(rfi_pixels, interpolation='nearest')
            plt.title('RFI pikseļi')
            plt.tight_layout()
            plt.show()

    return np.array(masks)

masks = flag(data, do_plot=False)
print(masks.shape)

with open(save_directory + 'LOFAR_subset_100_masks.pkl', 'wb') as f:
    pickle.dump(masks, f)





# ratiosum = 0.0
# ratiosumsq = 0.0
#
# vals = []
# for repeat in range(count):
#     for imgindex in range(8):
#         # Initialize data with random numbers
#         values = numpy.random.normal(0, 1, [nch, ntimes])
#         vals.append(values)
#         data.set_image_buffer(imgindex, values)
#
#     flags = strategy.run(data)
#     flagvalues = flags.get_buffer()
#     ratio = float(sum(sum(flagvalues))) / (nch*ntimes)
#     ratiosum += ratio
#     ratiosumsq += ratio*ratio
#     vals[0] = vals[0] * flagvalues
#
#

#
# print("Percentage flags (false-positive rate) on Gaussian data: " +
#     str(ratiosum * 100.0 / count) + "% +/- " +
#     str(numpy.sqrt(
#         (ratiosumsq/count - ratiosum*ratiosum / (count*count) )
#         ) * 100.0) )