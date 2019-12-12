# Data_Science_Project
Code for our implementation of thorax X-Ray image classification and localization, as part of the **Project Course in Data Science** for replicating and improving the results of the paper [ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases](https://arxiv.org/abs/1705.02315).

Four full information regarding the appraoches we took to tackle the problem and the results, please see our [report](report).

### To run the program:
There are a few arguments that can be used when running the program:

`--evaluate`: If used, the chosen model will be used for evaluation, that is, obtaining ROC curves, AUC numbers, or generating bounding boxes.

`--model_path`: The path to the model which is to be evaluated (only needed in the `--evaluate` flag is used).

`--use_comet`: If used, the experiment will be trakced in comet. Now only works for Moein's API key, but the experiment is public and everyone can view it.

`--save_checkpoints`: If used, the model checkpoints will be saved (at every epoch).

`--lr`: Determines the learning rate to be used

`--wdecay`: If set, weight decay is applied (see the code for more details).

`--max_epochs`: determines the maximum number of epochs to train the model

`--simple_lr_decay`: If used, a simple version of learning rate decay will be used (see the code for more details).

`----net_type`: Determines which network type should be used for training. It could be `unified_net`, `attention1`, `attention2`, or `attentionSE`. See the code for more details.
