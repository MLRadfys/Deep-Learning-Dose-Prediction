# Deep-Learning-Dose-Prediction

This repository contains the code used for our paper "Vmat dose prediction".

The repository is split into three different parts:

- **Preprocessing:** generates 2D and 2.5D (triplets) data from 3D volumes
- **Training:** Provides a densely connected UNet architecture that can be trained for VMAT dose predictions from triplets
- **NN search:** A script to find a similar dose distribution from a database using the mean squared error as a similarity metric

The different scripts should be run in the given numerical order.

## Before you start

### Setup

Clone the repository using Git Bash or the console  ``git clone https://ADRESS_TO_THE_GITHUB_REPOSITORY``. <br>

Create a virtual environment by typing ``virtualenv env_name`` into the console.

Once the virtual environment has been set up, it can be activated using ``source env_name/bin/activate``

Install the required libraries in the requirement.txt with  ``pip3 install -r requirements.txt``

Now all needed packages should have been installed into the virtual environment and the scripts can be run.

### Configuration File

All configuration parameters needed for training the model are stored in the ``config.ini`` file located in settings folder.
**Adjust the path to the dataset and the training/validation split csv file according to your setup.**

For the configuration file, it is possible to either give a folder name or a specific file.
If a .csv file is given, a single model will be trained. If a folder is given, a cross-validation is run, training 5 different models. The .csv files in the folder should follow the naming convention ``d_split_kx.csv``, where x is a number between 0-4.

## Preprocessing

The preprocessing script creates a 2D and a 2.5D dataset from given 3D volumes including RT Dose, CT images and RT structure sets.
To create the 3D volumes the MICE toolkit can be used, which is based on simpleITK. A free version is available here:

https://micetoolkit.com/

We also provid a screenshot of an example MICE workflow in the images folder of this repository. This workflow can be used to extract segmentation structures, dose matrices and CT images from DICOM files.

The expected folder structure for the 3D data is:
- Patient01
  - Dose
    - dose.npy   
  - masks
    - masks.npy 
  - CT   
    - CT.npy 
- Patient02
  - Dose
    - dose.npy   
  - masks
    - masks.npy 
  - CT   
    - CT.npy 
    
    ...

To create triplets (2.5D data) as used in our paper, run the ``create_dataset.py`` script using the console:
``python3 create_dataset.py -p PATH_TO_3D_DATA``
By running this script, a new folder is created, creating a 2D dataset, and a 2.5D dataset.
In addition, the script automatically creates csv files used for 5-fold cross validation training.

## Model

### Training

To train the model run the ``python3 train.py`` file. Depending on the configuration file, a single model is trained, or a cross-validation performed, resulting in 5 different models.
Model checkpoints are saved regulary, but this setting can be changed by modifying the ``CustomSaver.py`` script or building a self-defined checkpoint.
Training and validation processes are written to logfiles during training and can be examined using Tensorboard and can be examined during model training. For this, type ``tensorboard --logdir logs/fit`` in the console.

### Prediction

To predict a dose distribution using a trained model, run ``predict.py -id xyz`` from the console, where xyz is the subjectID for which a prediction should be performed.
To visualize the predicted dose runt ``predict.py -id xyz -v True``.

