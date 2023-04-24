# LOFAR_machine_learning

	<h4>LOFAR_machine_learning/unet<h4>

unet.py for training the model using LOFAR Long Term Archive images

unet_vsrc.py for training the model using VSRC spectograms

unet_load.py for loading already trained models from LOFAR_machine_learning/saved models

/saved_figs for graphs with predicted images

/check_gpu has scripts to check gpu availability for tensorflow and pytorch




	<h4>LOFAR_machine_learning/aoflag<h4>

flag_images.py generate masks for RFI pixels

/strategies/flag_images.py lua files for aoflagger strategies




	<h4>LOFAR_machine_learning/LOFAR	<h4>

/LOFAR subset 100/ pkl files for 100 LOFAR Long Term Archive images and generated AOFlagger masks (not on GIT)

/LOFAR subset 1000/ pkl files for 1000 LOFAR Long Term Archive images and generated AOFlagger masks (not on GIT)

save_subset.py for generating subsets of images from original LOFAR Long Term Archive images (total amount 72 thousand images)

check_subset.py for checking images in generated subsets

/VSRC_data/ holds LOFAR spectograms (not used anywhere at the moment)

