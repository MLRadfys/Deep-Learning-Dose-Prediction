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

All configuration parameters needed for training the model are stored in the ``config.ini`` file located in Training - settings folder.
Depending on where the training dataset and the cross validation files are stored, adjustments to the file paths are needed.


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

To create triplets as used in our paper, run the ``create_dataset.py`` script contained in the Preprocessing folder using the console:
``python3 create_dataset.py -p PATH_TO_3D_DATA``
By running this script, a new folder is created, creating a 2D dataset, and a 2.5D dataset.
In addition, the script automatically creates csv files used for 5-fold cross validation training.

## Model training

To train the model run ``python3 train.py`` in the Training folder.

