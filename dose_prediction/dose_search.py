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

import os
import argparse 
import numpy as np
import multiprocessing as mp
from natsort import natsorted
from keras.models import load_model
from skimage.measure import regionprops
from settings.ConfigLoader import ConfigLoader
from skimage.metrics import mean_squared_error as mse
from skimage.transform import AffineTransform, warp
#from skimage.measure import  compare_mse as mse /This has been replaced in the new skimage version

class Dose_search:

    def __init__(self):
        
        c = ConfigLoader(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'settings/config.ini'))
        self.db_patients_path = c.db_patients_path
        self.data_dir_test_patients = c.data_dir_test_patients
        self.stack_size = c.stack_size
        self.IMG_PX_SIZE = c.IMG_PX_SIZE
        self.inChannel = c.inChannel
        self.model_path = c.model_path

        os.environ["CUDA_VISIBLE_DEVICES"]= c.CUDA_device

    def get_triplet_masks(self, patID, data_dir_triplets):

            """

            Loads triplet data.

            Inputs:
                patID (str): patientID.
                data_dir_triplets (str): path to the triplet data.           
            
            Returns:
                mask_triplets (arr): Array of segmentation mask triplets.

            """
            
            masks_triplets = np.zeros((64,192,192,21))
            mask_triplet_files = natsorted([f for f in os.listdir(data_dir_triplets + '/masks/') if patID in f])
            
            for i in range(len(mask_triplet_files)):     
                masks_triplets[i,:,:,:] = np.load(data_dir_triplets + '/masks/' + mask_triplet_files[i])
            
            return masks_triplets

    def get_masks_center(self, masks):

            """

            Method to find the center slice (2D) of a given segmentation volume.

            Inputs:
                masks (arr): segmentation volume.            
            Returns:
                center (int); index of the segmentation volumes center.

            """

            indexes = []   
                        
            #get indicies where PTV mask is found
            for i, ma in enumerate(masks):
                if np.max(ma) == 1:
                    indexes.append(i)
            

            center = (indexes[-1] - indexes[0]) // 2 + indexes[0]  

            return center, indexes 

    def get_ptv_coordinates(self, ptv_center_slice):

            """

            Compute center coordinates of a 2D binary segmentation mask.

            Inputs:
                ptv_center_slice (arr): segmentation volume.            
            Returns:
                center (int); index of the segmentation volumes center.

            """

            prop = regionprops(np.asarray(ptv_center_slice, dtype='uint8'))
            center_coord = prop[0].centroid

            return center_coord
        
            

    def translate_image (self, image, vector):

        """

        Moves and image using Affine transformation

        Inputs:
            image (arr): 3D volume. 
            vector (arr): transformation vector           
        Returns:
            center (int); index of the segmentation volumes center.

        """

        transform = AffineTransform(translation=vector)

        translated_img = warp(image, transform, mode='wrap', preserve_range=True)

        return translated_img
    

    def compute_mse(self, pat, dose_prediction, ptv_center_coordinates_testPatient):

        """

        Matches PTV center coordinates between a test patient and a DB patient and computes the MSE between two dose distributions

        Inputs:
            pat(str: patient ID
            dose_prediction (arr): dose prediction for a test patient
            ptv_centr_coordinates_testPatient (tuple): PTV center coordinates

        Returns:
         mse (float): Mean squared error between two dose distributions

        """

        dose_distribution = np.zeros((self.stack_size, self.IMG_PX_SIZE, self.IMG_PX_SIZE))
        triplets = np.zeros((self.stack_size, self.IMG_PX_SIZE, self.IMG_PX_SIZE, self.inChannel))

        for j in range(self.stack_size):
            temp = np.load(self.db_patients_path + 'Dose_labels_center/Dose_labels_' + str(j) + '_' + pat + '.npy')       
            dose_distribution[j,:,:] = temp[:,:,0]
            triplets[j,:,:,:] = np.load(self.db_patients_path + 'masks/masks_' + str(j) + '_' + pat + '.npy')    


        dose_distribution = np.moveaxis(dose_distribution, 0, -1)


        #Get PTV center slice for the actual DB patient
        ptv_center, indexes = self.get_masks_center(triplets[:,:,:,9])

        #Get PTV center coordinates for the actual DB patient
        ptv_center_coordinates_dbPatient = self.get_ptv_coordinates(triplets[ptv_center,:,:,9])
        
        #Translate dose distributions to same corrdinates as testpatients
        dose_translated = self.translate_image(dose_distribution, (-(ptv_center_coordinates_testPatient[1] - ptv_center_coordinates_dbPatient[1]),  (ptv_center_coordinates_dbPatient[0] - ptv_center_coordinates_testPatient[0])))
            
        
        return mse(dose_prediction, dose_translated)

    def search_dose(self,patID, num_kernels):

        """

            Compute center coordinates of a 2D binary segmentation mask.

            Inputs:
                ptv_center_slice (arr): segmentation volume.            
            Returns:
                center (int); index of the segmentation volumes center.

        """

        model = load_model(self.model_path)


        masks_triplets_test = self.get_triplet_masks(patID, self.data_dir_test_patients)    

        #get the slice index of the PTV center
        ptv_center_test, indexes_test = self.get_masks_center(masks_triplets_test[:,:,:,9])

        #get center pixel coordinates for the PTV using the PTV 2D center slice
        ptv_center_coordinates_testPatient = self.get_ptv_coordinates(masks_triplets_test[ptv_center_test,:,:,9])

        #predict a dose distribution
        dose_predictions = model.predict(masks_triplets_test, batch_size = 2)


        dose_predictions_copy = dose_predictions[:,:,:,0].copy()

        dose_predictions_copy = np.moveaxis(dose_predictions_copy, 0,-1)


        
        db_patients_all = np.unique([f.replace('.npy','').split('_')[2] for f in os.listdir(self.db_patients_path + '/masks')])
        #db_patients_all = db_patients_all[:20]


        all_args = []
        for pat in db_patients_all:
            args = pat, dose_predictions_copy, ptv_center_coordinates_testPatient
            all_args.append(args)
        if num_kernels == 0:
            p = mp.Pool(mp.cpu_count())
        else:
            p = mp.Pool(num_kernels)

        mse_values = p.starmap(self.compute_mse, all_args)
        p.close()
        p.join()


            
        print('Nearest neighbor for test patient is:', db_patients_all[int(np.argmin(mse_values))])




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--subject_id', required=True, help='Subject ID used for prediction', type = str)
    parser.add_argument('-cpu','--num_kernels', required=False, default = 0, help='Number of CPU kernels used', type = int)
    args = parser.parse_args()
    
    ds = Dose_search()


    ds.search_dose(args.subject_id, args.num_kernels)
    