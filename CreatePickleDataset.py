import pickle
import glob
import os
import numpy as np
from tqdm import tqdm
from datasets import ImageDataset

#arr1 = np.ones((100,100,3))
#print(arr1)


#filename = 'arr1.pkl'

#with open(filename,mode='wb') as f:
#    pickle.dump(arr1,f)

#with open(filename, mode= 'rb') as f:
#    arr1 = pickle.load(f)

path = "resources/Dataset"


def create_dir (index):

    dataset_path = os.path.join('resources','Pickle_Dataset',str(index))

    try:
        Flag = False
        os.mkdir(dataset_path, )
    except:
        Flag = True

    return Flag



def store (input_array, known_array, target_array, image_array, index):

    dataset_path = os.path.join('resources','Pickle_Dataset',str(index))

    array_list = [input_array, known_array, target_array, image_array, index]
    filenames = ['input_array.pkl', 'known_array.pkl', 'target_array.pkl', 'image_array.pkl', 'index.pkl']
    
    for i in range(5):

        with open(os.path.join(dataset_path,filenames[i]), mode= 'wb') as f:
            pickle.dump(array_list[i], f)



image_paths = sorted(glob.glob(os.path.join(path, "**", "*.jpg"), recursive=True))
dataset = ImageDataset(path)

for index_ in tqdm(range(len(image_paths))):

    Flag = create_dir(index_)
    #print('step 1',Flag)

    if not Flag:
        input_array, known_array, target_array, image_array, index = dataset.__getitem__(index_)
        #print('step 2')

        store(input_array, known_array, target_array, image_array, index)
        #print('step 3')

    #print(index_,'/',len(image_paths))





