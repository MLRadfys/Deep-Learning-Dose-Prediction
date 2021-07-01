#==============================================================================#
#  Author:       Michael Lempart                                                #
#  Copyright:    2021, Department of Radiation Physics, Lund University                                       #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#

#import libraries
import os
import sys
import csv 
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
from natsort import natsorted
from sklearn.model_selection import KFold
from settings.ConfigLoader import ConfigLoader
from sklearn.model_selection import train_test_split
  

class Preprocessor:

    def __init__(self):

        """

        Class initializer

        Inputs:
            None
        Returns:
            None
        
        """
        try:
            self.config = ConfigLoader(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'settings/config.ini'))
        except:
            print("The config.ini file could not be loaded, please check the file path...")
            sys.exit()
        

    def center_crop(self, img, new_width=None, new_height=None):  
        
        """

        Center crops image data to a specific height and width

        Inputs:
            img (array): Image data
            new_width (int): new width that the image should be cropped to
            new_height (int): new height that the image should be cropped to
        Returns:
            center_cropped_img ( array): center cropped image

        """

        width = img.shape[1]
        height = img.shape[2]

        if new_width is None:
            new_width = min(width, height)

        if new_height is None:
            new_height = min(width, height)

        left = int(np.ceil((width - new_width) / 2))
        right = width - int(np.floor((width - new_width) / 2))

        top = int(np.ceil((height - new_height) / 2))
        bottom = height - int(np.floor((height - new_height) / 2))

        
        center_cropped_img = img[:, top:bottom, left:right]
    

        return center_cropped_img


    def normalize_HU(self, images, min_bound = -1000.0, max_bound = 400.0):

        """

        Truncates and normalizes HU values

        Inputs:
            images (array): Images in the HU range
        Returns:
            image_normalized ( array): Normalized images

        """

        MIN_BOUND = min_bound
        MAX_BOUND = max_bound

        image_normalized = (images - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image_normalized[image_normalized>1] = 1.
        image_normalized[image_normalized<0] = 0.

        return image_normalized



    def create_dataset2D(self, data_dir, new_path, normalization = 42.7):

        """

        Creates a 2D dataset from 3D numpy arrays

        Inputs:
            data_dir (str): path to preprocessed 3D numpy arrays 
            new_path (str): save dir for preprocessed data
        Returns:
            None

        """
        patients = os.listdir(data_dir)

        patients = patients[:5]
    
        for pat in tqdm(patients): 
                
            mask_dir = new_path + '/' + pat + '/masks/'
            dose_dir = new_path + '/' + pat + '/Dose/'
            CT_dir = new_path + '/' + pat + '/CT/'
            
            os.makedirs(mask_dir, exist_ok=True)         
            os.makedirs(dose_dir, exist_ok=True)        
            os.makedirs(CT_dir, exist_ok=True)        
        

            masks = np.load(data_dir +  pat + '/masks/masks.npy')
            #Correct all masks to a value of 1
            masks[masks > 0] = 1
        
            dose = np.load(data_dir +  pat + '/Dose/dose.npy')   
            
            CT = np.load(data_dir +  pat + '/CT/CT.npy') 
            
            masks = self.center_crop(masks, self.config.IMG_PX_SIZE, self.config.IMG_PX_SIZE)
            CT = self.center_crop(CT, self.config.IMG_PX_SIZE,self.config.IMG_PX_SIZE)
            dose = self.center_crop(dose, self.config.IMG_PX_SIZE, self.config.IMG_PX_SIZE)
            
            ptv_masks = masks[:,:,:,1]

            indexes = []
                    
            #get indicies where PTV mask is found
            for i, mask in enumerate(ptv_masks):
                if np.max(mask) == 1:
                        indexes.append(i)
            
            ptv_center = (indexes[-1] - indexes[0]) // 2 + indexes[0]  
            
            masks = masks[ptv_center-32:ptv_center+32, :, :, :]              

            dose = dose[ptv_center-32:ptv_center+32, :, :, :]   
            
            CT = CT[ptv_center-32:ptv_center+32, :, :, :] 
            CT_normalized = self.normalize_HU(CT)  

            #Normalize Dose
            min_ = 0            
            max_ = normalization          

            d_normalized = (dose - min_) / (max_ - min_)   
            

            
            for i in range(len(masks)):

                img_m_0 = nib.Nifti1Image(masks[i,:,:,:], np.eye(4))                

                nib.save(img_m_0, mask_dir + '/masks_{}_{}.nii.gz'.format(i, pat))      
                        
                img_m2 = nib.Nifti1Image(d_normalized[i,:,:,:], np.eye(4))
            
                nib.save(img_m2, dose_dir + '/Dose_labels_{}_{}.nii.gz'.format(i, pat))  

                img_m3 = nib.Nifti1Image(CT_normalized[i,:,:,:], np.eye(4))
            
                nib.save(img_m3, CT_dir + '/CT_{}_{}.nii.gz'.format(i, pat))  
        

        
    def create_tripletsFromMask(self, dataset_dir_2D, new_path, masks, pat):    

        """

        Creates image triplets from a 2D dataset

        Inputs:
            dataset_dir_2D (str): path to preprocessed 2D dataset 
            new_path (str): save dir for preprocessed data
        Returns:
            None

        """
    
        masks = natsorted(masks)

        dose_files = os.listdir(dataset_dir_2D + pat + '/Dose/')
        dose_files = natsorted(dose_files)
        
        CT_files = os.listdir(dataset_dir_2D + pat + '/CT/')
        CT_files = natsorted(CT_files)
                
        for i in tqdm(range(len(masks))):

            #CENTER DOSE
            dose_center = nib.load(dataset_dir_2D + pat +  '/Dose/' + dose_files[i]).get_data()    

            #CENTER CT
            CT_center = nib.load(dataset_dir_2D + pat +  '/CT/' + CT_files[i]).get_data()    
                    
            if i == 0:

                CT_left = nib.load(dataset_dir_2D + pat +  '/CT/' + CT_files[i]).get_data()   
                CT_right = nib.load(dataset_dir_2D + pat +  '/CT/' + CT_files[i+1]).get_data()
            
                m1 = nib.load(dataset_dir_2D + pat +  '/masks/' + masks[i]).get_data()
                m2 = nib.load(dataset_dir_2D + pat +  '/masks/' + masks[i]).get_data()
                m3 = nib.load(dataset_dir_2D + pat +  '/masks/' + masks[i+1]).get_data()
            
            elif i == len(masks)-1:
                CT_left = nib.load(dataset_dir_2D + pat +  '/CT/' + CT_files[i-1]).get_data()  
                CT_right = nib.load(dataset_dir_2D + pat +  '/CT/' + CT_files[i]).get_data() 
            
                m1 = nib.load(dataset_dir_2D + pat +  '/masks/' + masks[i-1]).get_data()
                m2 =nib.load(dataset_dir_2D + pat +  '/masks/' + masks[i]).get_data()
                m3 = nib.load(dataset_dir_2D + pat +  '/masks/' + masks[i]).get_data()
            
            else:
                CT_left = nib.load(dataset_dir_2D + pat +  '/CT/' + CT_files[i-1]).get_data() 
                CT_right = nib.load(dataset_dir_2D + pat +  '/CT/' + CT_files[i+1]).get_data()  
                
                m1 = nib.load(dataset_dir_2D + pat +  '/masks/' + masks[i-1]).get_data()
                m2 = nib.load(dataset_dir_2D + pat +  '/masks/' + masks[i]).get_data()
                m3 = nib.load(dataset_dir_2D + pat +  '/masks/' + masks[i+1]).get_data()

            
            all_channels_with_CT = np.zeros((self.config.IMG_PX_SIZE,self.config.IMG_PX_SIZE,self.config.inChannel))

            all_channels_with_CT[:,:,0] = CT_left[:,:,0] 
            all_channels_with_CT[:,:,1] = m1[:,:,0]  
            all_channels_with_CT[:,:,2] = m1[:,:,1]          
            all_channels_with_CT[:,:,3] = m1[:,:,2]  
            all_channels_with_CT[:,:,4] = m1[:,:,3]  
            all_channels_with_CT[:,:,5] = m1[:,:,4]  
            all_channels_with_CT[:,:,6] = m1[:,:,5]  
            all_channels_with_CT[:,:,7] = CT_center[:,:,0]  
            all_channels_with_CT[:,:,8] = m2[:,:,0]  
            all_channels_with_CT[:,:,9] = m2[:,:,1]  
            all_channels_with_CT[:,:,10] = m2[:,:,2]  
            all_channels_with_CT[:,:,11] = m2[:,:,3]  
            all_channels_with_CT[:,:,12] = m2[:,:,4]  
            all_channels_with_CT[:,:,13] = m2[:,:,5]  
            all_channels_with_CT[:,:,14] = CT_right[:,:,0]
            all_channels_with_CT[:,:,15] = m3[:,:,0]  
            all_channels_with_CT[:,:,16] = m3[:,:,1]  
            all_channels_with_CT[:,:,17] = m3[:,:,2]  
            all_channels_with_CT[:,:,18] = m3[:,:,3]  
            all_channels_with_CT[:,:,19] = m3[:,:,4]  
            all_channels_with_CT[:,:,20] = m3[:,:,5]  
            
            np.save(new_path + '/masks/masks_{}_{}.npy'.format(i,pat),all_channels_with_CT)           
            np.save(new_path + '/Dose_labels_center/Dose_labels_{}_{}.npy'.format(i,pat), dose_center)  


    def split_dataset(self, data_dir):          

        """

        Splits the dataset into 5 folds for cross-validation training

        Inputs:
            data_dir (str): path to preprocessed 3D dataset            
        Returns:
            None

        """

        training_files = os.listdir(data_dir)


        kf = KFold(n_splits=5, random_state=None, shuffle=False)

        k = 0
        for train_index, val_index in kf.split(training_files):

            X_train = np.asarray(training_files)[train_index.astype(int)]
            y_train = X_train
            X_val = np.asarray(training_files)[val_index.astype(int)]
            y_val = X_val
        

            dataset = []
            dataset_test = []
            dataset_train = []
            labels_train = []
            dataset_val = []
            labels_val = []
            dataset_test = []
            labels_test = []

            for j, x in enumerate(X_train):
                files = os.listdir(data_dir + x + '/masks/')            
                for i in range(len(files)):
                    f_temp = files[i][5:-7]            
                    dataset.append([f_temp, 'training'])             
                            
            for j, x in enumerate(X_val):
                files = os.listdir(data_dir + x + '/masks/')        
                for i in range(len(files)):
                    f_temp = files[i][5:-7]                    
                    dataset.append([f_temp, 'validation'])         
                
            with open('d_split_k{}.csv'.format(k), mode='w') as file_:
                writer = csv.writer(file_, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for i in range(len(dataset)):
                    writer.writerow([dataset[i][0], dataset[i][1]])

            k += 1     
            
    
    def make_triplets(self, data_dir, new_path):


        """

        Creates image triplets from a 2D dataset

        Inputs:
            data_dir (str): path to preprocessed 2D dataset 
            new_path (str): save dir for preprocessed data
        Returns:
            None

        """
        
    
        os.makedirs(new_path + '/masks',exist_ok=True) 
        os.makedirs(new_path + '/Dose_labels_center', exist_ok=True) 
        
            
        folders = os.listdir(data_dir)

        
        for pat in tqdm(folders):    

            masks_temp = os.listdir(data_dir + pat + '/masks/') 

            self.create_tripletsFromMask(data_dir, new_path, masks_temp, pat) 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True, help='Path to 3D numpy arrays')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))

    data_dir = args.path

    p = Preprocessor()

    new_data_dir_2D = os.path.join(script_dir, 'dataset/preprocessed/2D/')
    #Create a 2D dataset
    print('--> creating 2D data...')
    p.create_dataset2D(data_dir, new_data_dir_2D, normalization = 42.7)

    #Create triplets
    print('--> creating 2.5D data...')
    new_data_dir = os.path.join(script_dir, 'dataset/preprocessed/triplets/')
    p.make_triplets(new_data_dir_2D,new_data_dir)
    
    #Create k-cross validation splits
    print('--> splitting dataset...')
    p.split_dataset(new_data_dir_2D)
    print('processing done...')


