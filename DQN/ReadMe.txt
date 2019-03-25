The files in this repository are enough to begin training an A3C model. 
Tensorpack package must been installed. 

Examples training command;
$ python DQN-Train.py --env MsPacman-v0 --algo DQN 

Model files for each env have been included in the 'train_log'. They are the model files from 1,500,000 steps which were used to generate the data used in this project. This was the training cut off for
the agent, so it is very well trained and can perform well in these environments. 

The agents may be tested in these games by running the command;
$ python DQN-Train.py --env MsPacman-v0 --algo DQN --load ./train_log/DQN-MsPacman-v0/model-1500000 --task play (to play MsPacman for instance)

By default 'render=true', this will show the agent playing the game. If running from Google Colab and not locally then this will need to be changed to false.
It is changed in the common.py file, line 45, in the 'play_n_episodes' method arguments. 

DRL.ipynb is the iPython notebook which should be run on Google Colab to train using the cloud. 