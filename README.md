# Deep-Learning-Dose-Prediction

This repository contains the code used for our paper "Vmat dose prediction".

The repository is split into three different parts:

- Preprocessing: generates 2D and 2.5D (triplets) data from 3D volumes
- Training: Provides a densely connected UNet architecture that can be trained for VMAT dose predictions from triplets
- NN search: A script to find a similar dose distribution from a database using the mean squared error as a similarity metric



Create a virtual environment with virtualenv env_name

Activate the newly created virtual environment using source env_name/bin/activate

Install the required libraries in the requirement.txt --> pip3 install -r requirements.txt

