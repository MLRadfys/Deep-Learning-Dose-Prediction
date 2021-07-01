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
import keras
from keras.utils import multi_gpu_model
import numpy as np
from unet_utils.data_generator import DataGenerator2D
from unet_utils.tensor_board_logger import TrainValTensorBoard
from unet_utils.CustomSaver import CustomSaver
from unet_utils.utils import get_data
from neural_network.DenseUnet import DenseUNet
from sklearn.utils import shuffle
from settings.ConfigLoader import ConfigLoader




def train():

  
    os.environ["CUDA_VISIBLE_DEVICES"]="2" # first gpu
    os.environ["CUDA_VISIBLE_DEVICES"]="3" # second gpu


    c = ConfigLoader(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'settings/config.ini'))
    
    cross_validation_files = c.csv_file
            
    data_dir_training = data_dir_val = c.data_dir

    input_dim = (c.IMG_PX_SIZE, c.IMG_PX_SIZE,c.inChannel) 

    #Set up dictionaries for the data generator
    params = {'dim': (c.IMG_PX_SIZE,c.IMG_PX_SIZE),
                            'batch_size' : c.batch_size,
                            'n_channels': c.inChannel,
                            'shuffle': True,
                            'augment': c.augmentation}  

    params_val = {'dim': (c.IMG_PX_SIZE,c.IMG_PX_SIZE),
                            'batch_size' : c.batch_size,
                            'n_channels': c.inChannel,
                            'shuffle': False,
                            'augment': c.augmentation}  

    
    folds = [0,1,2,3,4]

    #Run a single split
    if c.csv_file.endswith('.csv'):
        
        X_train, y_train, X_val, y_val = get_data(c.csv_file)
    
        model_name = c.name + '_BS_{}_Dropout_{}_L2_{}'.format(c.batch_size, c.dropout, c.L2)
        save_dir = 'Model/' + model_name

        unet = DenseUNet(input_dim, dropout_rate = c.dropout, l2 = c.L2, lr = c.lr)
        model = unet.build_DenseUNet()

        #if c.n_gpus > 1:    
            #model = multi_gpu_model(model, gpus=c.n_gpus)
        
        os.makedirs(save_dir, exist_ok=True)

        #Set up Model checkpoints
        checkpoints = CustomSaver(save_dir + 'Checkpoints_' + model_name)
        
        #Set up training and validation data generators    
        dataGen_training = DataGenerator2D(X_train, y_train, data_dir_training, dose_label='center', **params)
        dataGen_validation = DataGenerator2D(X_val, y_val, data_dir_val,dose_label='center', **params_val)
        
        #Set up callbacks
        callbacks = [TrainValTensorBoard('{}_training'.format(model_name), '{}_validation'.format(model_name), write_graph=True), checkpoints]

        #Train the model
        model.fit(dataGen_training, validation_data =dataGen_validation, verbose = 1, epochs= c.epochs , callbacks = callbacks, workers = c.workers)
    
    else:
        
        #Run k-fold cross validation
        for k in folds:

            csv_file = os.path.join(c.csv_file, 'd_split_k{}.csv'.format(k))

            X_train, y_train, X_val, y_val = get_data(csv_file)  
        
            model_name = c.name + '_BS_{}_Dropout_{}_L2_{}_fold_{}'.format(c.batch_size, c.dropout, c.L2, k)
            save_dir = 'Model/' + model_name

            unet = DenseUNet(input_dim, dropout_rate = c.dropout, l2 = c.L2, lr = c.lr)
            model = unet.build_DenseUNet()

            #if c.n_gpus > 1:    
                #model = multi_gpu_model(model, gpus=c.n_gpus)
            
            os.makedirs(save_dir, exist_ok=True)

            #Set up Model checkpoints
            checkpoints = CustomSaver(save_dir + 'Checkpoints_' + model_name)
            
            #Set up training and validation data generators    
            dataGen_training = DataGenerator2D(X_train, y_train, data_dir_training, dose_label='center', **params)
            dataGen_validation = DataGenerator2D(X_val, y_val, data_dir_val,dose_label='center', **params_val)
            
            #Set up callbacks
            callbacks = [TrainValTensorBoard('{}_training'.format(model_name), '{}_validation'.format(model_name), write_graph=True), checkpoints]

            #Train the model
            model.fit(dataGen_training, validation_data =dataGen_validation, verbose = 1, epochs= c.epochs , callbacks = callbacks, workers = c.workers)

if __name__ == "__main__":
    
    train()
