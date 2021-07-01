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

from configparser import ConfigParser

class ConfigLoader:

    """

    The config loader class can be used to load a custom config file.

    """

    def __init__(self, config_path = 'config.ini'):

        """

        Class initializer

        Args:
            config_path (str): Path of the configuration file

        Returns:
            None

        """

        self.config_path = config_path
        #[PATH]
        self.data_dir = None
        self.csv_file = None
        self.db_patients_path = None
        self.save_dir = None
        #[IMAGE_DATA]
        self.IMG_PX_SIZE = None
        self.inChannel = None
        self.stack_size = None
        #[NETWORK]
        self.name = None
        self.workers = None
        #[TRAINING]
        self.CUDA_device = None
        self.n_gpus = None
        self.lr = None
        self.epochs = None
        self.L2 = None
        self.dropout = None
        self.augmentation = None
        #[PREDICTION]
        self.model_path = None
        self.data_dir_test_patients = None

        self.load_config()

        
    def load_config(self):

        try:

            print('loading configuration file...')

            config = ConfigParser()

            config.read(self.config_path)

            print('CONFIGURATION SECTIONS:')
            print(config.sections())

            self.data_dir = str(config.get('PATH','data_dir'))
            self.csv_file = str(config.get('PATH','csv_file'))
            self.db_patients_path = str(config.get('PATH','db_patients_path'))
            self.save_dir = str(config.get('PATH','save_dir'))

            self.IMG_PX_SIZE = int(config.get('IMAGE_DATA','IMG_PX_SIZE')) 
            self.inChannel = int(config.get('IMAGE_DATA','inChannel'))
            self.stack_size = int(config.get('IMAGE_DATA','stack_size'))

            self.name = str(config.get('NETWORK','name'))
            self.workers = int(config.get('NETWORK','workers'))

            self.CUDA_device = str(config.get('TRAINING','CUDA_device'))
            self.n_gpus = int(config.get('TRAINING','n_gpus'))
            self.lr = float(config.get('TRAINING','lr'))
            self.epochs = int(config.get('TRAINING','epochs'))
            self.batch_size = int(config.get('TRAINING','batch_size'))
            self.L2 = float(config.get('TRAINING','L2'))
            self.augmentation = bool(config.get('TRAINING','augmentation'))
            try:
                self.dropout = float(config.get('TRAINING','dropout'))
            except:
                self.dropout = str(config.get('TRAINING','dropout'))
            
            self.model_path = str(config.get('PREDICTION','model_path'))
            self.data_dir_test_patients = str(config.get('PREDICTION','data_dir_test_patients'))

        except FileNotFoundError:            
            print("No config file found. Make sure that the config.ini file is located in the root folder...")
        
        except ValueError:
            print("One or several configuration tags do not match the configuration file standard....") 

