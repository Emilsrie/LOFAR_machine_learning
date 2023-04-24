# LOFAR_machine_learning

### LOFAR_machine_learning/unet<br>

unet.py for training the model using LOFAR Long Term Archive images</br>
unet_vsrc.py for training the model using VSRC spectograms</br>
unet_load.py for loading already trained models from LOFAR_machine_learning/saved models</br>
/saved_figs for graphs with predicted images</br>
/check_gpu has scripts to check gpu availability for tensorflow and pytorch</br>



### LOFAR_machine_learning/aoflag

flag_images.py generate masks for RFI pixels</br>
/strategies/flag_images.py lua files for aoflagger strategies</br>


### LOFAR_machine_learning/LOFAR

/LOFAR subset 100/ pkl files for 100 LOFAR Long Term Archive images and generated AOFlagger masks (not on GIT)</br>
/LOFAR subset 1000/ pkl files for 1000 LOFAR Long Term Archive images and generated AOFlagger masks (not on GIT)</br>
/VSRC_data/ VSRC LOFAR generated image examples (Currently not )
save_subset.py for generating subsets of images from original LOFAR Long Term Archive images (total amount 72 thousand images)</br>
check_subset.py for checking images in generated subsets</br>
</br>
Download dataset from: https://zenodo.org/record/6724065#.ZEbiCnZBybi and use save_subset.py to generate necessary subsets**</br>

## NOT ON GIT
LOFAR data (bst format) before image creation</br>
VSRC created code for spetogram creation</br>
Generated spectogram images and prepared dataset</br>

