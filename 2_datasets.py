import os
import glob
import utils
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset



class ImageDataset (Dataset):

    def __init__(self, file_paths:list = None):
        """
        takes a list of file_paths as input 
        """
        if file_paths != None and type(file_paths) != list:
            raise ValueError ('file_paths is not of type list')

        self.files = sorted(file_paths)
        