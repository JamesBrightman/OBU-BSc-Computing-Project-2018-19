# -*- coding: utf-8 -*-
# File: common.py
# Author: Yuxin Wu
import multiprocessing
import random
import time
from six.moves import queue
from tqdm import tqdm
import matplotlib.pyplot as plt
import statistics
import os

from collections import Counter
from tensorpack.callbacks import Callback
from tensorpack.utils import logger
from tensorpack.utils.concurrency import ShareSessionThread, StoppableThread
from tensorpack.utils.stats import StatCounter
from tensorpack.utils.utils import get_tqdm_kwargs
from PIL import Image, ImageDraw, ImageFont


def play_one_episode(env, func, render=True):
    def predict(s):
        """
        Map from observation to action, with 0.01 greedy.
        """
        act = func(s[None, :, :, :])[0][0].argmax()
        if random.random() < 0.01:
            spc = env.action_space
            act = spc.sample()
        return act

    ob = env.reset()
    sum_r = 0
    while True:
        act = predict(ob)
        ob, r, isOver, info = env.step(act)
        if render:
            env.render()
        sum_r += r
        if isOver:
            return sum_r


def play_n_episodes(ENV_NAME, player, predfunc, nr, render=False):
#JB ---- Rewritten entire method
    logger.info("Start Playing ... ")
    #Initialising variables 
    test_game_rewards = []
    test_game_rewards_average = []
    best_score = -100

    for k in range(nr): #starting playing though number of episodes
        score = play_one_episode(player, predfunc, render=render)   #returns game score
        test_game_rewards.append(score) #add score to array of scores
        test_game_rewards_average.append(sum(test_game_rewards)) #add sum of game rewards to average score array

        #print verbose information about training progress
        print("\n# --------------------------------------------------------- #")
        print("GAMES PLAYED SO FAR: {}".format(k+1))
        print("GAME SCORE: {}".format(score))
        if k > 0:
            print("AVERAGE SCORE: {:5f}".format(sum(test_game_rewards_average)/(k+1)))   
        if sum(test_game_rewards) > best_score:
              best_score = sum(test_game_rewards)
        print("BEST SCORE: {}".format(best_score))          
        print("Game Finished. Restarting Game...")
        print("# --------------------------------------------------------- #\n")
        test_game_rewards.clear() #remove all game scores from game reward array 
        
    fig = plt.figure()  #create new figure
    box_plot_data=test_game_rewards_average    #initiliase variable
    plt.boxplot(box_plot_data, vert=False, showmeans=True) #create boxplot from data
    plt.xlabel('Score', fontsize=18)    #X axis label
    plt.ylabel(' ', fontsize=18)    #Y axis label
    plt.title(ENV_NAME + " - A3C")  #title
    plt.savefig("boxPlot.png")  #save boxplot to png
    #calculating statistics from saved data
    mean = statistics.mean(test_game_rewards_average)
    median = statistics.median(test_game_rewards_average)
    c = Counter(test_game_rewards_average)
    mode = c.most_common(2)

    img = Image.new('RGB', (400, 100), color = (255, 255, 255)) #create new blank image
    font = ImageFont.truetype("arial.ttf", 15)  #set font
    d = ImageDraw.Draw(img)
    d.text((10,10), "Mean = {}\nMedian = {}\nMode = {}".format(mean,median,mode), fill=(0,0,0), font=font)  #adding stats text
    img.save('stats.png')   #saving image with stat data

    box = Image.open("boxPlot.png") #open boxplot
    stats = Image.open("stats.png") #open stats
    area = (105,425)    #define area
    box.paste(stats, area)  #paste stats picture onto boxplot picture
    box.save(ENV_NAME+"-boxPlot.png")   #save new compound picture

    os.remove("boxPlot.png")    #remove original box-plot picture
    os.remove("stats.png")  #remove stat data picture

    with open(ENV_NAME+ "-data" + ".csv", "w") as out_file: #create CSV 
        for i in range(len(test_game_rewards_average)): #read in game scores
            out_string = ""
            out_string += str(test_game_rewards_average[i])
            out_string += ","
            out_string += "\n"
            out_file.write(out_string)  #add game score string to CSV
# JB ----
def eval_with_funcs(predictors, nr_eval, get_player_fn, verbose=False):
    """
    Args:
        predictors ([PredictorBase])
    """
    class Worker(StoppableThread, ShareSessionThread):
        def __init__(self, func, queue):
            super(Worker, self).__init__()
            self._func = func
            self.q = queue

        def func(self, *args, **kwargs):
            if self.stopped():
                raise RuntimeError("stopped!")
            return self._func(*args, **kwargs)

        def run(self):
            with self.default_sess():
                player = get_player_fn(train=False)
                while not self.stopped():
                    try:
                        score = play_one_episode(player, self.func)
                    except RuntimeError:
                        return
                    self.queue_put_stoppable(self.q, score)

    q = queue.Queue()
    threads = [Worker(f, q) for f in predictors]

    for k in threads:
        k.start()
        time.sleep(0.1)  # avoid simulator bugs
    stat = StatCounter()

    def fetch():
        r = q.get()
        stat.feed(r)
        if verbose:
            logger.info("Score: {}".format(r))

    for _ in tqdm(range(nr_eval), **get_tqdm_kwargs()):
        fetch()
    # waiting is necessary, otherwise the estimated mean score is biased
    logger.info("Waiting for all the workers to finish the last run...")
    for k in threads:
        k.stop()
    for k in threads:
        k.join()
    while q.qsize():
        fetch()

    if stat.count > 0:
        return (stat.average, stat.max)
    return (0, 0)


def eval_model_multithread(pred, nr_eval, get_player_fn):
    """
    Args:
        pred (OfflinePredictor): state -> [#action]
    """
    NR_PROC = min(multiprocessing.cpu_count() // 2, 8)
    with pred.sess.as_default():
        mean, max = eval_with_funcs(
            [pred] * NR_PROC, nr_eval,
            get_player_fn, verbose=True)
    logger.info("Average Score: {}; Max Score: {}".format(mean, max))


class Evaluator(Callback):
    def __init__(self, nr_eval, input_names, output_names, get_player_fn):
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names
        self.get_player_fn = get_player_fn

    def _setup_graph(self):
        NR_PROC = min(multiprocessing.cpu_count() // 2, 20)
        self.pred_funcs = [self.trainer.get_predictor(
            self.input_names, self.output_names)] * NR_PROC

    def _trigger(self):
        t = time.time()
        mean, max = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn)
        t = time.time() - t
        if t > 10 * 60:  # eval takes too long
            self.eval_episode = int(self.eval_episode * 0.94)
        self.trainer.monitors.put_scalar('mean_score', mean)
        self.trainer.monitors.put_scalar('max_score', max)
