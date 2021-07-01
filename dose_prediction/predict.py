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
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from settings.ConfigLoader import ConfigLoader
import argparse

def predict(subject_id, verbose = False):

    config = ConfigLoader(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'settings/config.ini'))

    os.environ["CUDA_VISIBLE_DEVICES"]= config.CUDA_device
    
    model_path = config.model_path
    
    model = load_model(config.model_path)    
        
    m = np.zeros((config.stack_size,config.IMG_PX_SIZE,config.IMG_PX_SIZE,config.inChannel))

    for i in range(config.stack_size):
        m[i,:,:,:] = np.load(config.data_dir + 'masks/masks_' + str(i) + '_' + subject_id + '.npy')

    dose_predictions = model.predict(m)

    if verbose:
        recon_figure = np.zeros((config.IMG_PX_SIZE * 8,config.IMG_PX_SIZE * 8))     

        idx = 0       
        for i in range(8):
            for j in range(8):         
                recon_figure[i * config.IMG_PX_SIZE : (i+1) * config.IMG_PX_SIZE,
                                j * config.IMG_PX_SIZE : (j+1) * config.IMG_PX_SIZE] = dose_predictions[idx,:,:,0]*42.7   
                                
                idx += 1

        plt.imshow(recon_figure, cmap = 'jet', vmin = 0.0, vmax = 45.0)
        plt.axis('off')
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--subject_id', required=True, help='Subject ID used for prediction', type = str)
    parser.add_argument('-v', '--verbose',  required=False, default = False, help='Shows a plot of the prediction if True', type = bool)
    args = parser.parse_args()
    
    predict(args.subject_id, args.verbose)