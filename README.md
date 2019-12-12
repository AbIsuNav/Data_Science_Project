# Data_Science_Project
Code for our implementation of thorax X-Ray image classification and localization, as part of the **Project Course in Data Science** for replicating and improving the results of the paper [ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases](https://arxiv.org/abs/1705.02315).

### To run the program:
There are a few arguments that can be used when running the program:

`--evaluate`

`--model_path`

`--use_comet`: if used, the experiment will be trakced in comet. Now only works for Moein's API key, but the experiment is public and everyone can view it.

`--save_checkpoints`: if used, the model checkpoints will be saved (at every epoch).

`--lr`: determines the learning rate to be used

`--wdecay`:

`--max_epochs`: determines the maximum number of epochs to train the model

`--simple_lr_decay`
