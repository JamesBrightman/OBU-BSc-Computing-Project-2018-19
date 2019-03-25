#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Yuxin Wu

import argparse
import numpy as np
import os
import cv2
import gym
import tensorflow as tf

from tensorpack import *

from atari_wrapper import FireResetEnv, FrameStack, LimitLength, MapState
from common import Evaluator, eval_model_multithread, play_n_episodes
from DQNModel import Model as DQNModel
from expreplay import ExpReplay

BATCH_SIZE = 64 # Larger than 32 to promote GPU usage
IMAGE_SIZE = (84, 84)
IMAGE_CHANNEL = None  # 3 in gym and 1 in our own wrapper
FRAME_HISTORY = 4   # Number of stacked frames
ACTION_REPEAT = 4   # aka FRAME_SKIP
UPDATE_FREQ = 4

GAMMA = 0.99 # Discount factor

MEMORY_SIZE = 1e5
# will consume at least 1e6 * 84 * 84 bytes == 6.6G memory.
INIT_MEMORY_SIZE = MEMORY_SIZE // 20
STEPS_PER_EPOCH = 100000 // UPDATE_FREQ  # each epoch is 100k played frames
EVAL_EPISODE = 50 # number of episodes performance is evaluated across

NUM_ACTIONS = None
USE_GYM = False
ENV_NAME = None
METHOD = None

#JB ---
INIT_EXP=1.0 #exploration value (epsilon) 
START_EP=1 #starting epoch - for continued training


def resize_keepdims(im, size):
    # Opencv's resize remove the extra dimension for grayscale images.
    # We add it back.
    ret = cv2.resize(im, size)
    if im.ndim == 3 and ret.ndim == 2:
        ret = ret[:, :, np.newaxis]
    return ret


def get_player(viz=False, train=False):
    if USE_GYM:
        env = gym.make(ENV_NAME)
    else:
        from atari import AtariPlayer
        env = AtariPlayer(ENV_NAME, frame_skip=ACTION_REPEAT, viz=viz,
                          live_lost_as_eoe=train, max_num_frames=60000)
    env = FireResetEnv(env)
    env = MapState(env, lambda im: resize_keepdims(im, IMAGE_SIZE))
    if not train:
        # in training, history is taken care of in expreplay buffer
        env = FrameStack(env, FRAME_HISTORY)
    if train and USE_GYM:
        env = LimitLength(env, 60000) # max 60k timesteps
    return env


class Model(DQNModel):
    def __init__(self):
        super(Model, self).__init__(IMAGE_SIZE, IMAGE_CHANNEL, FRAME_HISTORY, METHOD, NUM_ACTIONS, GAMMA)

    def _get_DQN_prediction(self, image):
        image = image / 255.0
        with argscope(Conv2D, activation=lambda x: PReLU('prelu', x), use_bias=True):
            l = (LinearWrap(image)
                 # Nature architecture
                 .Conv2D('conv0', 32, 8, strides=4)
                 .Conv2D('conv1', 64, 4, strides=2)
                 .Conv2D('conv2', 64, 3)
                 .FullyConnected('fc0', 512)
                 .tf.nn.leaky_relu(alpha=0.01)())
        if self.method != 'Dueling':
            Q = FullyConnected('fct', l, self.num_actions)
        else:
            # Dueling DQN - NOT USED in this project
            V = FullyConnected('fctV', l, 1)
            As = FullyConnected('fctA', l, self.num_actions)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))
        return tf.identity(Q, name='Qvalue')


def get_config():
    expreplay = ExpReplay(
        predictor_io_names=(['state'], ['Qvalue']),
        player=get_player(train=True),
        state_shape=IMAGE_SIZE + (IMAGE_CHANNEL,),
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        init_exploration=INIT_EXP,      
        update_frequency=UPDATE_FREQ,
        history_len=FRAME_HISTORY
    )

    return TrainConfig(
        data=QueueInput(expreplay),
        model=Model(),
        callbacks=[
            ModelSaver(),
            PeriodicTrigger(
                RunOp(DQNModel.update_target_param, verbose=True),
                every_k_steps=10000 // UPDATE_FREQ),    # update target network every 10k steps
            expreplay,
            ScheduledHyperParamSetter('learning_rate',
                                      [(60, 4e-4), (100, 2e-4), (500, 5e-5)]),
            ScheduledHyperParamSetter(
                ObjAttrParam(expreplay, 'exploration'),
                [(0, 1), (10, 0.1), (320, 0.01)],   #Jb - 1->0.1 in the first million frames (250k steps)
                interp='linear'),
            PeriodicTrigger(Evaluator(
                EVAL_EPISODE, ['state'], ['Qvalue'], get_player),
                every_k_epochs=10),
            HumanHyperParamSetter('learning_rate'),
        ],
        steps_per_epoch=STEPS_PER_EPOCH,
        starting_epoch=START_EP,
        max_epoch=60, #JB - max 1.5 mill steps
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--task', help='task to perform',
                        choices=['play', 'eval', 'train'], default='train')
    parser.add_argument('--env', required=True,
                        help='either an atari rom file (that ends with .bin) or a gym atari environment name')
    parser.add_argument('--algo', help='algorithm',
                        choices=['DQN', 'Double', 'Dueling'], default='Double')
#JB ----
    parser.add_argument('--exp', help='Exploration value')
    parser.add_argument('--ep', help='Starting Epoch')
    args = parser.parse_args()
#   ----
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    ENV_NAME = args.env
    USE_GYM = not ENV_NAME.endswith('.bin')
    IMAGE_CHANNEL = 3 if USE_GYM else 1
    METHOD = args.algo

    NUM_ACTIONS = get_player().action_space.n
    logger.info("ENV: {}, Num Actions: {}".format(ENV_NAME, NUM_ACTIONS))

#JB ----
    if args.exp:
        INIT_EXP = float(args.exp)  # use exploration if gives as an argument

    if args.ep:
        START_EP = int(args.ep)    # use epoch if gives as an argument
#   ----
    if args.task != 'train':
        assert args.load is not None
        pred = OfflinePredictor(PredictConfig(
            model=Model(),
            session_init=get_model_loader(args.load),
            input_names=['state'],
            output_names=['Qvalue']))
        if args.task == 'play':
            play_n_episodes(ENV_NAME, get_player(viz=0.01), pred, 88) #sample size 88 due to 0.95 statistical power
        elif args.task == 'eval':
            eval_model_multithread(pred, EVAL_EPISODE, get_player)
    else:
        logger.set_logger_dir(
            os.path.join('train_log', 'DQN-{}'.format(
                os.path.basename(ENV_NAME).split('.')[0])))
        config = get_config()
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(config, SimpleTrainer())
