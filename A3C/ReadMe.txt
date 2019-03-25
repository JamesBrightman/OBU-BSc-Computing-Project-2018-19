The files in this repository are enough to begin training an A3C model. 
Tensorpack package must been installed. 

Examples training command;
$ python train-atari.py --env Boxing-v0 --gpu 0

Model files for each env have been included in the 'train_log'. They are the model files from 1,500,000 steps which were used to generate the data used in this project. This was the training cut off for
the agent, so it is very well trained and can perform well in these environments. 

The agents may be tested in these games by running the command;
$ python train-atari.py --env Boxing-v0 --load ./train_log/train-atari-Boxing-v0/model-1500000 --task play (to play Boxing for instance)

By default 'render=true', this will show the agent playing the game. If running from Google Colab and not locally then this will need to be changed to false.
This is changed on line 297 in 'train-atari.py', in the arguments to calling the 'plan_n_episodes' method. 

DRL.ipynb is the iPython notebook which should be run on Google Colab to train using the cloud. 