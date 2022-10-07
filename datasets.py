#IMPORTS
import os
import glob
import utils
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset
##################################
"""
This files contains the Dataset class used for the ML project
The Dataset class needs two functions to work properly
1. __getitem__
    returns an Images in the perfect way for the CNN as input and also the index
2. __len__
    returns the size of the Dataset or the amount of Images in the Dataset folder
"""

#DATASET CLASS
class ImageDataset(Dataset):
    def __init__(self, path):
        """
        This class loads prepares the Images for Usage in the Network.\n
        • standradices the images\n
        • resizes the images to 100x100\n
        • creates offset and spacing\n\n

        returns:\n
        1. input_array -> the image as array with offset and spacing dtype=\n
        2. known_array -> a mask showing if a pixel value is provided or not dtype=\n
        3. target_array -> array of the missing pixel values dtype=\n
        4. image_array -> the complete image as array dtype=\n
        5. index -> the index of the image dtype=\n

        """
        self.path = path
        self.image_paths = sorted(glob.glob(os.path.join(path, "**", "*.jpg"), recursive=True))

        #normalize
        self.analyzer = utils.ImageStandardizer(self.path)
        self.mean, self.std = self.analyzer.analyze_images()

    def __getitem__(self, index: int):
        """Here we have to define a method to get 1 sample
        
        __getitem__() should take one argument: the index of the sample to get

        return -> input_array, known_array, target_array, image_array, index
        """
        #normalize
        image_data = self.analyzer.get_standard_image(index)
        #select the chosen image and resize it 
        image_array = utils.image_shaper(image_data, 100)
        #random int
        offset = tuple(np.random.randint(0,9,2))
        spacing = tuple(np.random.randint(2,7,2))
        #offset and spacing
        input_array, known_array, target_array = utils.image_poker(image_array, offset, spacing)
        #return the value and the ID
        return input_array, known_array, target_array, image_array, index

    def __len__(self):
        """ Optional: Here we can define the number of samples in our dataset
        
        __len__() should take no arguments and return the number of samples in
        our dataset
        """
        n_samples = len(self.image_paths)

        return n_samples

class PickleImageDataset (Dataset):
    def __init__(self, path):
        """
        This class loads and prepares for Usage in the Network.\n
        
        returns:\n
        1. input_array -> the image as array with offset and spacing\n
        2. known_array -> a mask showing if a pixel value is provided or not\n
        3. target_array -> array of the missing pixel values\n
        4. image_array -> the complete image as array\n
        5. index -> the index of the image\n

        """
        self.path = path
        self.image_paths = sorted(glob.glob(os.path.join(path, "**", "image_array.pkl"), recursive=True))

        #normalize
        self.analyzer = utils.ImageStandardizer(self.path, pickle = True)
        self.mean, self.std = self.analyzer.analyze_images_pkl()

    def __getitem__(self, index: int):
        """Here we have to define a method to get 1 sample
        
        __getitem__() should take one argument: the index of the sample to get

        return -> input_array, known_array, target_array, image_array, index
        """

        self.image_paths = sorted(glob.glob(os.path.join(self.path, str(index), "*.pkl"), recursive=True))
        image_arrays = []
        for image_path in self.image_paths:

            with open(image_path, mode= 'rb') as f:
                x = pkl.load(f)

            image_arrays.append(x)

        image = image_arrays[0]
        ids = image_arrays[1]
        inputs = image_arrays[2]
        knowns = image_arrays[3]
        targets = image_arrays[4]

        print(f'targets:{targets}')
        
        return image, ids, inputs, knowns, targets

        
    def __len__(self):
        """ Optional: Here we can define the number of samples in our dataset
        
        __len__() should take no arguments and return the number of samples in
        our dataset
        """
        paths = glob.glob(os.path.join(self.path, "**", "*.pkl"), recursive=True)
        n_samples = len(paths)/5

        return int(n_samples)