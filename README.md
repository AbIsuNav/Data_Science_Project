# Data_Science_Project
X-Ray image classification and localization

### To run the program:
There are a few arguments that can be used when running the program:
--use_comet: if used, the experiment will be trakced in comet. Now only works for Moein's API key, but the experiment is public and everyone can view it.

--save_checkpoints: if used, the model checkpoints will be saved (at every epoch).

--no_crop: if used, the 256x256 images will not be cropped to 224

--lr: determines the learning rate to be used

--max_epoxhs: determines the maximum number of epochs to train the model
