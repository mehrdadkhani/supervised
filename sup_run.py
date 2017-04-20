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

from mahimahiInterface import RLMahimahiInterface
from supervised_tcp_learner import *

def action_minmax(a, lower, upper):
    ret = a
    if ret > upper:
        ret = upper
    if ret < lower:
        ret = lower

    return ret


def main():

    # Initialize Environment Interface
    intf = RLMahimahiInterface()
    arg = str(sys.argv)
    intf.ConnectToMahimahi(ip=sys.argv[1])

    intf.SetRLState(1)

    # Initialize Actor (Tensorflow)
    tf_sess = tf.Session()

    dump_time_interval = 2.0
    tf_sess.run(tf.initialize_all_variables())

    # Initialize Variables
    # throughput_dequeue = 0
    MinimumExecutorUpdateInterval = 0.02
    avg_time_interval = MinimumExecutorUpdateInterval

    prev_prob = 0.0


    last_dump_ts = time()

    actor_update_counter = 0

    ep = 0



    while True:
        # Start Time
        start_time = time()

        # Get Observations from Environment
        ob = intf.GetState(dump=False)

        if ob[0] > 0:
            ingress = ob[10] * 8 / avg_time_interval / 1000000.0
        else:
            ingress = 0.0

        new_s = [prev_prob, ingress]


        new_prob = random.uniform(0.0,0.6)
        intf.SetDropRate(new_prob)


        # Generate Experience
        experience = {'state': np.reshape(prev_state, (2,)),
                      'next_state': np.reshape(new_s, (2,)),
                      'dropprob': new_prob}

        prev_prob = new_prob
        prev_state = new_s

        # Publish New Experience
        redis_server.publish('Experience', pickle.dumps(experience))


        # Dump Infomation
        if time() - last_dump_ts > dump_time_interval:
            last_dump_ts = time()
            print('EXECUTOR ', ep, ' : Last Drop Prob: ', new_prob, ' Avg Time Interval: ', avg_time_interval)
            ep += 1

        # Time Management
        time_passed = time() - start_time
        if time_passed < MinimumExecutorUpdateInterval:
            sleep(MinimumExecutorUpdateInterval - time_passed)

        time_passed = time() - start_time
        avg_time_interval = avg_time_interval * 0.75 + time_passed * 0.25


if __name__ == '__main__':
    main()
