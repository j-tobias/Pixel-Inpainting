import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Tuple
from matplotlib import pyplot as plt

import torch
import torch.utils.data
from torchvision.transforms import transforms




def mse ():
    pass

class ImageStandardizer:

    def __init__ (self, input_dir: str, pickle: boolean = False, filepath_list: list = []):
        """
        Input \n
        • input_dir: which is the path to an input directory. This can be an absolute or relative path. \n
        • pickle: states if the input files are .pkl files or .jpg files \n
        • filepath_list: instead a list to the files can be given \n \n



        • Scan this input directory recursively for files ending in .jpg.\n
        • Raise a ValueError if there are no .jpg files.\n
        • Transform all paths to absolute paths and sort them alphabetically in ascending order.\n
        • Store the sorted absolute file paths in an attribute self.files.\n
        • Create an attribute self.mean with value None.\n
        • Create an attribute self.std with value None.        
        """

        if filepath_list == []:
            if pickle:
                #search recursively for image_array.pkl files an
                jpgs_input_dir = glob.glob(os.path.join(input_dir, '**','image_array.pkl'), recursive=True)
            else:
                #search recursively for *.jpg files an
                jpgs_input_dir = glob.glob(os.path.join(input_dir, '**','*.jpg'), recursive=True)
        else:
            self.filepath_list = filepath_list
            jpgs_input_dir = filepath_list

        #check if files were found or not -> if not raise Error
        if jpgs_input_dir == []:
            self.file = []
            raise ValueError("There seem to be no .jpg files in this dir")
        
        if filepath_list == []:
            #create the absolute paths
            abs_path_jpgs = []
            for jpg_file in tqdm(jpgs_input_dir):
                abs_path_jpgs.append(os.path.abspath(jpg_file))

            #sort the paths alphabetically in ascending order
            self.files = sorted(abs_path_jpgs)
        else:
            self.files = filepath_list


        ####just for Testing###
        #for file in self.files:
        #    print(file)

        #create two attributes with value None
        self.mean = None
        self.std = None

    def analyze_images (self):
        """
        • Take no additional arguments. \n
        • Compute the means and standard deviations for each color channel of all images in the list self.files. Each mean and standard deviation will thus have three entries: one for the red (R), one for the green (G) and one for the blue channel (B). \n
        • Store the average over these RGB means of all images in the attribute self.mean (global RGB mean). This value should be a 1D numpy array of datatype np.float64 and with shape (3,). \n
        • Store the average over these RGB standard deviations of all images in the attribute self.std (global RGB standard deviation). This value should be a 1D numpy array of datatype np.float64 and with shape (3,). \n
        • Return the tuple (self.mean, self.std). \n
        """


        self.mean = np.zeros(3, dtype=np.float64)
        self.std = np.zeros(3, dtype=np.float64)


        #iterating thourgh all files
        for image in self.files:

            #open the current Image to get the data
            with Image.open(image) as image:
                Image_mean = ImageStat.Stat(image).mean
                Image_std  = ImageStat.Stat(image).stddev

                self.mean += Image_mean
                self.std += Image_std

        self.mean /= len(self.files)
        self.std /= len(self.files)

        return (self.mean, self.std)

    def analyze_images_pkl (self):
        """
        • Take no additional arguments. \n
        • Compute the means and standard deviations for each color channel of all images in the list self.files. Each mean and standard deviation will thus have three entries: one for the red (R), one for the green (G) and one for the blue channel (B). \n
        • Store the average over these RGB means of all images in the attribute self.mean (global RGB mean). This value should be a 1D numpy array of datatype np.float64 and with shape (3,). \n
        • Store the average over these RGB standard deviations of all images in the attribute self.std (global RGB standard deviation). This value should be a 1D numpy array of datatype np.float64 and with shape (3,). \n
        • Return the tuple (self.mean, self.std). \n
        """
        self.mean = np.zeros(3, dtype=np.float64)
        self.std = np.zeros(3, dtype=np.float64)

        #iterating thourgh all files
        for image_array_path in tqdm(self.files):

            #open the current Picklefile to get the data
            with open(image_array_path, 'rb') as f:
                image_array = pkl.load(f)

                self.mean += np.mean(image_array)
                self.std += np.std(image_array)

        self.mean /= len(self.files)
        self.std /= len(self.files)

        return (self.mean, self.std)


    def get_standardized_images(self):

        #check if global mean or global std is None -> in case Raise Error
        if self.mean is None or self.std is None:
            raise ValueError("Mean or std is None or both")


        #iterate through all files
        for image in self.files:
            
            #open the Image and transform it into an array with dtype = np.float32
            image_data = np.asarray(Image.open(image), dtype= np.float32)

            #subtract the mean
            image_data -= self.mean

            #normalize by using the std
            image_data = image_data / self.std

            
            yield np.asarray(image_data, dtype=np.float32)
    
    def get_standard_image(self, index):

        #check if global mean or global std is None -> in case Raise Error
        if self.mean is None or self.std is None:
            raise ValueError("Mean or std is None or both")

        image = self.files[index]

        #open the Image and transform it into an array with dtype = np.float32
        image_data = np.asarray(Image.open(image), dtype= np.float32)

        #subtract the mean
        image_data -= self.mean

        #normalize by using the std
        image_data = image_data / self.std

        return np.asarray(image_data, dtype=np.float32)