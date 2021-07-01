import numpy as np
import keras
import os
import pydicom
import scipy
import pickle
from keras.utils import np_utils
import nibabel as nib
import imgaug as ia
from imgaug import augmenters as iaa

       
            
class DataGenerator2D(keras.utils.Sequence):

    def __init__(self, list_IDs, labels, data_dir, batch_size = 16, dim=(192,192), n_channels = 21, shuffle = True, augment = True, dose_label = 'center'):

        """

        Creates a Data generator which reads numpy arrays from patient folders.
        Inputs:
            list_IDs (list): List with patient folder names
            labels (list): List with labels (for CAE, list_IDs = labels)
            n_channels (int): number of image input channels
            dim (tuple): Dimensions of the input image (H,W,Depth)
            data_dir (str): Data directory of the folders
            shuffle (bool): If True, List_IDs get shuffled after each epoch
            random_state (int): random state used for shuffling in order to get repeatable results   
            normalize (bool): Normalizes dosegrid values between 0 and 42.6Gy    
        Returns:
            None
            
        """

        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.dim = dim
        self.data_dir = data_dir
        self.shuffle = shuffle            
        self.on_epoch_end()        
        self.augment = augment              
        self.dose_label = dose_label
        

    

    def __len__(self):


        """
        
        Determines the number of batches per epoch.
        Inputs:
            None
        Returns:        
            batches (int): Number of batches
        
        """

        batches = int(np.floor(len(self.list_IDs) / self.batch_size))        
        return batches
    
    def __getitem__(self, index):

        """
        
        Generates one batch of data.
        Inputs:
            index (int): determines the start index for the batch in the given training data
        Returns:
            X, y (array): returns 3D volumes and corresponding labels (ground truth)

        """
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #indexes = index
       
        #Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        labels_temp = [self.labels[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp, labels_temp)


        if self.augment == True:
            X,y = self.augmentor(X,y)

        return X, y       
     
   

    def on_epoch_end(self):

        """

        Updates and shuffles indexes after each epoch.
        Inputs:
            None
        Return:
            None

        """
       
        self.indexes = np.arange(len(self.list_IDs))        
      
        if self.shuffle == True:            
            np.random.shuffle(self.indexes)

   
    def __data_generation(self, list_IDs_temp, labels_temp, triplets=False):

        """

        Generates data containing batch_size samples
        Inputs:
            list_IDs_temp (list): list of indexes used to create on batch of data
        Returns:
            volumes (array): 3D volumes used for training

        """
      
       
            
        masks_batch = np.empty((len(list_IDs_temp), self.dim[0],self.dim[1],self.n_channels))
        doseGrid_labels = np.empty((len(list_IDs_temp), self.dim[0],self.dim[1]))
       

        for j, (f, pat_folder) in enumerate(zip(list_IDs_temp, labels_temp)):
            
            path = self.data_dir 

            masks = np.load(path + '/masks/masks' + f + '.npy')    

            
            masks_batch[j,:,:,:] = masks 
            
              
            if self.dose_label == 'center':         
                dose_center = np.load(path + '/Dose_labels_center/Dose_labels' + f + '.npy')  
            elif self.dose_label == 'left':
                dose_center = np.load(path + '/Dose_labels_left/Dose_labels' + f + '.npy')  
            elif self.dose_label == 'right':
                dose_center = np.load(path + '/Dose_labels_right/Dose_labels' + f + '.npy')  
           
            try:
                doseGrid_labels[j,:,:] = dose_center[:,:,0]                  
            except:
                doseGrid_labels[j,:,:] = dose_center[:,:]   
       

        doseGrid_labels = np.reshape(doseGrid_labels, (doseGrid_labels.shape[0], doseGrid_labels.shape[1], doseGrid_labels.shape[2],1))                  
            
        
        return masks_batch, doseGrid_labels     


    def augmentor(self, images, labels):
        'Apply data augmentation'
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        iaa.Sequential()
        seq = iaa.Sequential(
                [
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                sometimes(iaa.Affine(
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, #translate the image +- 10% in x and y
                    rotate=(-5, 5),  # rotate by -5 to +5 degrees
                    mode=ia.ALL,
                ))   
                ], random_order=True
        )
        seq_det = seq.to_deterministic()
        images_aug = seq_det.augment_images(images)
        labels_aug = seq_det.augment_images(labels)
        
        return images_aug, labels_aug                        
