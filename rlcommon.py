from subprocess import Popen, PIPE
from numpy import uint32
from numpy import int32
from numpy import uint64
from numpy import int64
from time import sleep, time
import random
import os
import socket
import sys

import tensorflow as tf
import tflearn
import numpy as np
import pickle
import redis

# ********** Replay Buffer **************
# Random seed
RANDOM_SEED = 44

# Size of Replay Buffer
BUFFER_SIZE = 6000

MINIBATCH_SIZE = 64



# ********** Tensorflow ******************
# Dimension of state (input)
in_dim = 6

# Dimension of action (output)
out_dim = 1


# How long do we save the neural network model
nn_model_save_interval = 1000000

MAX_THRPUT = 20.0 # 20Mbps

LEARNING_RATE = 0.001#0.00001


SUMMARY_DIR = './results/rl_dpg_train'


# ************ Control *******************
IsRLEnabled = 1 ################### 1

SamplingTime = 0.02 # 20ms

MinimumTrainerUpdateInterval = 0.1 # 100ms


dump_time_interval = 2 # 2 seconds


