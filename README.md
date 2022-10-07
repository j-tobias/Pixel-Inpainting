# Pixel Inpainting AI Project
This ML project contains a CNN that is trained to inpaint missing Pixel Values.
The input images used are randomly collected photos of my study batch. Those images are downscaled to 100x100 and are of the type 'RGB'



### Usage
This project can easily be run if everything is setup properly (packages + versions)
in this case run -> main.py

Another Usage idea could be to input own images where the pixels not have been removed but spreaded, such that
gaps appeared. Those gaps could then be inpainted and afterwards would result in an higher resolution of the photo


### Structure
PIXEL-INPAINTING
| - architecture.py
|       Classes and functions for network architectures
| - datasets.py
|       Dataset classes and dataset helper functions
| - utils.py
|       Utility functions and classes.
| - main.py
|       Main file. In this case also includes training and evaluation routines.
| - working_config.json
|       An example configuration file. Can also be done via command line arguments to main.py.
| - resources
|       | - plots
|       |       Plots of the Performence of the Training Process and the Perfromence of the Model
|       | - tensorboard
|       |       log files of the tensorboard
|       | - best_model.pt
|       |       weights of the model with the best performence
|       | - Dataset
|       |       | - 000
|       |       |       containing the collected images (roughly 100)
|       |       | - 001
|       |       |       containing the collected images (roughly 100)
|       |       ...(all together are around 29410 images)


### Dependencies
Library         |Version
------------------------------
- glob          |0.7
- torch         |1.11.0
- numpy         |1.21.5
- PIL           |9.0.1
- tqdm          |4.64.0
- matplotlib    |3.5.1
- python        |3.9.8

