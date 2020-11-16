# Zencastr Machine Learning Homework

Welcome to Zencastr's Machine Learning Engineering homework. In this repository, you will implement some basic data pre-processing and model training scripts. The model we're training today is [MOSNet](https://arxiv.org/abs/1904.08352), a 2D CNN that generates mean opinion scores (scores from 1 to 5 relating to the quality of the speech signal) based on a fullband audio spectrogram. Training data is provided, and is from the [TCD-VoIP MOS dataset](http://www.mee.tcd.ie/~sigmedia/Resources/TCD-VoIP). It should be small enough for you to complete an epoch comfortably on a laptop.

## What you'll do

You have three scripts here, `model.py`, `utils.py`, and `train.py`.

`model.py` is already generated, so you don't need to do anything there, but you should get yourself acquainted.

`utils.py` is where you have to implement some basic pre-processing, including spectrogram extraction. Look for `TODO` comments to find places to work. Once your methods are complete, you should be able to run `utils.py` directly to extract features and save them in the HDF5 binary format. Besides pre-processing, there are also various helper functions in `utils.py`, include the data generator function that will feed your model training.

Finally, `train.py` is where you'll build your training script. Here, you'll have to set hyperparameters, create training and validation sets, and compile the model, among other things. There are some `TODO` comments guiding you along. Once you complete this script, complete at least one epoch of training. While you won't be tested on how your model performs, please consider ways you could improve the models performance. After training, save your model to the output folder.

## What to return

Please return a snapshot of your repository after completing at least one epoch of training. Please include a requirements.txt file with all the packages you used.
