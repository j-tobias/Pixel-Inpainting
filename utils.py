######IMPORTS
import os
import glob
from xmlrpc.client import boolean
import tqdm
import math
import torch
import numpy as np
import dill as pkl
from PIL import Image
from tqdm import tqdm
from PIL import ImageStat
import matplotlib as plt
from torchvision import transforms
from torch.utils.data import DataLoader
##################################
"""
This files contains Utility functions and classes.
Each function is described more detailed at it's place 

"""
######CODE
#ImageStandardizer
"""
A helper class to perform some overall calculations on the Dataset
such as mean, std and normalizing
"""
class ImageStandardizer:

    def __init__ (self, input_dir: str, pickle: boolean = False):
        """
        Take one keyword argument input_dir (string), which is the path to an input directory. This can be an absolute or relative path. \n
        • Scan this input directory recursively for files ending in .jpg.\n
        • Raise a ValueError if there are no .jpg files.\n
        • Transform all paths to absolute paths and sort them alphabetically in ascending order.\n
        • Store the sorted absolute file paths in an attribute self.files.\n
        • Create an attribute self.mean with value None.\n
        • Create an attribute self.std with value None.        
        """

        if pickle:
            #search recursively for image_array.pkl files an
            jpgs_input_dir = glob.glob(os.path.join(input_dir, '**','image_array.pkl'), recursive=True)
        else:
            #search recursively for *.jpg files an
            jpgs_input_dir = glob.glob(os.path.join(input_dir, '**','*.jpg'), recursive=True)

        #check if files were found or not -> if not raise Error
        if jpgs_input_dir == []:
            self.file = []
            raise ValueError("There seem to be no .jpg files in this dir")
        
        #create the absolute paths
        abs_path_jpgs = []
        for jpg_file in jpgs_input_dir:
            abs_path_jpgs.append(os.path.abspath(jpg_file))

        #sort the paths alphabetically in ascending order
        self.files = sorted(abs_path_jpgs)

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
        for image_array_path in self.files:

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

#_get_offset
"""
predefined for image_poker
sets the defined offset to 0 like a border at the top and left
"""

def _get_offset (array: np.ndarray, offset: np.ndarray):

    offset_y = offset[1]
    offset_x = offset[0]

    for channel in range(np.shape(array)[0]):
        array_channel = array[channel]
        for row in range(np.shape(array_channel)[0]):
            array_row = array_channel[row]
            for value in range(len(array_row)):
                if row < offset_y or value < offset_x:
                    array[channel,row,value] = 0

    return array
#_get_spacing
"""
predefined for image_poker
pokes holes into the image
"""
def _get_spacing (array: np.ndarray , spacing: tuple, offset: tuple):
    spacing_y = spacing[1]
    spacing_x = spacing[0]
    offset_y = offset[1]
    offset_x = offset[0]

    for channel in range(np.shape(array)[0]):
        counter_empties_y = spacing_y
        counter_empties_x = spacing_x
        array_channel = array[channel]

        for row in range(np.shape(array_channel)[0]):
            if row > offset_y and counter_empties_y <= spacing_y and counter_empties_y > 0: #(offset_y - row) == 0 or
                array[channel,row] = 0
                counter_empties_y -= 1
            elif row > offset_y:
                counter_empties_y = spacing_y

        for value in range(np.shape(array_channel)[1]):
            if value > offset_x and counter_empties_x <= spacing_x and counter_empties_x > 0: #(offset_x - row) == 0 or 
                array[channel,:,value] = 0
                counter_empties_x -= 1
            elif value > offset_x:
                counter_empties_x = spacing_x

    return array
#image_poker
"""
takes an Image and returns it with the the specified offset and spacing\n
meaningly it 'delets' intermediate pixel values 
"""
def image_poker(image_array, offset, spacing) -> tuple:
    """
    image_array = the image as array (H,W,3)\n
    offset = distance from border to first pixel on Left and Top\n
    spacing = Space between two pixels\n
    
    returns (input_array, known_array, target_array)
    """
    #test if image_array is a numpy array
    # if isinstance(np.ndarray, type(image_array)): #numpy.ndarray
    if str(type(image_array)) != "<class 'numpy.ndarray'>":
        raise TypeError("image_array is not a Numpy Array")


    #try if it is possible to call all three dimensions
    try:
        np.shape(image_array)[0]
        np.shape(image_array)[1]
        np.shape(image_array)[2]
    except:
        raise NotImplementedError("image_array has less than 3 dimensions - 1")
    
    #test if image_array is not a 3D array or 3rd dimension is not equal to 3
    if np.shape(image_array)[2] != 3:
        raise NotImplementedError("image_array is not a 3D array or 3rd dimension is not 3")

    #try if it is a 3 dimansional array
    try:
        IndexError_flag = False
        np.shape(image_array)[3]
    except IndexError:
        IndexError_flag = True

    if IndexError_flag == False:
        raise NotImplementedError("image_array has more than 3 dimensions - 2")

    #test convertion to int
    try:
        int(offset[0])
        int(offset[1])
        int(spacing[0])
        int(spacing[1])
    except:
        raise ValueError("values in offest and/or spacing cannot be converted to int")

    #test value range of offset
    if offset[0] < 0 or offset[1] < 0 or offset[0] > 32 or offset[1] > 32:
        raise ValueError("values of offset are either lager than 32 or smaller than 0")
    
    #test value range of spacing
    if spacing[0] < 2 or spacing[1] < 2 or spacing[0] > 8 or spacing[1] > 8:
        raise ValueError("values of spacing are either lager than 8 or smaller than 2")

    
    #test remaining pixels
    X = np.shape(image_array)[1]
    Y = np.shape(image_array)[0]
    off_x = offset[0]
    off_y = offset[1]
    spac_x = spacing[0]
    spac_y = spacing[1]
    num_pixels = math.ceil((math.ceil(((X - off_x)*(Y - off_y))/ (spac_y))) / (spac_x))
    
    if num_pixels < 144:
        raise ValueError("The number of the remaining known image pixels would be smaller than 144")


    #####Testing Finished
    spacing_new = (spacing[0]-1,spacing[1]-1)

    
    input_array = np.transpose(image_array, (2, 0, 1))
    known_array = np.ones_like(input_array)
    known_array = _get_offset(known_array, offset)
    known_array = _get_spacing(known_array, spacing_new, offset)
    target_array = []

    for channel in range(np.shape(input_array)[0]):
        array_channel = input_array[channel]
        for row in range(np.shape(array_channel)[0]):
            array_row = array_channel[row]
            for value in range(len(array_row)):
                if known_array[channel,row,value] == 0:
                    target_array.append(input_array[channel,row,value])
                    input_array[channel,row,value] = 0  

    target_array = np.asarray(target_array)

    return (input_array, known_array, target_array) 

#Image Shaper
"""
resizes the images to specified shape or intuitivly to 100x100
"""
def image_shaper (image: np.ndarray = None, image_shape:int = 100) -> np.ndarray:
    resize_transforms = transforms.Compose([
        transforms.Resize(size=image_shape),
        transforms.CenterCrop(size=(image_shape, image_shape))
    ])
    image = Image.fromarray(image, mode= 'RGB')
    image = resize_transforms(image)
    return np.asarray(image)

#Result Plotting
"""
creates a plot of the current model stat, performance
"""
def plot(inputs, targets, predictions, path, update):
    """Plotting the inputs, targets and predictions to file `path`"""
    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    
    for i in range(len(inputs)):
        for ax, data, title in zip(axes, [inputs, targets, predictions], ["Input", "Target", "Prediction"]):
            ax.clear()
            ax.set_title(title)
            ax.imshow(data[i, 0], cmap="gray", interpolation="none")
            ax.set_axis_off()
        fig.savefig(os.path.join(path, f"{update:07d}_{i:02d}.png"), dpi=100)
    
    plt.close(fig)

#Evaluation


def mse(outputs, targets):
    mse_loss = torch.nn.MSELoss()

    # Getting just the unknown cells.
    # masked_outputs = outputs[known_arrays < 1]

    # Returning the MSE loss.
    # return mse_loss(masked_outputs, targets)

    print(outputs.size(), targets.size())

    return mse_loss(outputs, targets)
"""
def mse(input_images, predictions, n_samples) -> np.ndarray:
    #get ground truth for each sample
    groundtruths = []
    for image in  input_images:
        pass
    #get error for each sample
    errors = []
    for groundtruth, prediction in groundtruths, predictions:
        errors.append(groundtruth - prediction)
    #square the losses
    squared_losses = []
    for error in errors:
        squared_losses.append(error**error)
    #add the losses and devide by n_samples
    loss = sum(squared_losses)/n_samples
    return loss"""

def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, loss_fn, device: torch.device):
    """Function for evaluation of a model `model` on the data in `dataloader` on device `device`,
    using the specified `loss_fn` loss function"""
    model.eval()
    # We will accumulate the mean loss in variable `loss`
    loss = 0
    with torch.no_grad():  # We do not need gradients for evaluation
        # Loop over all samples in `dataloader`
        for data in tqdm(dataloader, desc="scoring", position=0):
            # Get a sample and move inputs and targets to device
            inputs, targets, file_names = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Get outputs of the specified model
            outputs = model(inputs)
            
            # Here we could clamp the outputs to the minimum and maximum values of inputs for better performance
            
            # Add the current loss, which is the mean loss over all minibatch samples
            # (unless explicitly otherwise specified when creating the loss function!)
            loss += loss_fn(outputs, targets).item()
    # Get final mean loss by dividing by the number of minibatch iterations (which
    # we summed up in the above loop)
    loss /= len(dataloader)
    model.train()
    return loss

def denormalize_image(image, mean, std):
    # Converting from Torch Tensor to numpy array for calculations.
    image *= std
    image += mean
    return image