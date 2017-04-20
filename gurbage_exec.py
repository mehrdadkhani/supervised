from rlcommon import *
from mahimahiInterface import RLMahimahiInterface
from ddpg import *


def action_minmax(a, lower, upper):
    ret = a
    if ret > upper:
        ret = upper
    if ret < lower:
        ret = lower

    return ret


def main():
    # Initialize Redis for IPC
    redis_server = redis.StrictRedis(host='localhost', port=6379, db=0)
    redis_pubsub = redis_server.pubsub()
    redis_pubsub.subscribe('Actor')

    # Initialize Environment Interface
    intf = RLMahimahiInterface()
    arg = str(sys.argv)
    intf.ConnectToMahimahi(ip=sys.argv[1])

    intf.SetRLState(IsRLEnabled)

    # Initialize Actor (Tensorflow)
    tf_sess = tf.Session()

    actor = ActorNetwork(tf_sess, state_dim, action_dim, action_bound, ACTOR_LEARNING_RATE,
                         TAU)  # *********** action_bound is replace with 1

    tf_sess.run(tf.initialize_all_variables())

    # Initialize Variables
    throughput_dequeue = 0
    avg_time_interval = MinimumExecutorUpdateInterval

    prev_prob = 0.1
    prev_actor_action = 0
    prev_state = np.zeros(state_dim)

    ob_sliding_window_size = 384
    ob_sliding_window = np.zeros(ob_sliding_window_size, dtype=float)
    normalized_ob_wind = np.zeros(ob_sliding_window_size, dtype=float)

    q_hist = np.zeros(5, dtype=float)
    th_hist = np.zeros(5, dtype=float)
    p_hist = np.zeros(5, dtype=float)

    last_dump_ts = time()

    actor_update_counter = 0

    ep = 0
    qdelay_mov_avg = 0
    thrput_mov_avg = 0
    mov_avg_drop = 0.0
    mov_avg_th = 0.0
    mov_avg_q = 0.0
    cnt = 0

    lockflag = False
    lockertime = time()
    outage_flag = False
    outage_start = time()
    # Read the channel information

    prev_prev_p = 0.0
    prev_q = 0.0
    prev_th = 0.0

    # channel = [line.rstrip('\n') for line in open('channel.txt')]
    channel_time = time()

    while True:
        # Start Time
        start_time = time()
        # print (time() - channel_time)/300.0
        # action_bound = min(max(0.2, (time() - channel_time)/300.0), 1)
        # print action_bound

        # Get Observations from Environment
        ob = intf.GetState(dump=False)
        throughput_dequeue = ob[7] / 0.02 * 8 / 1000000  # Unit: Mbps
        queue_delay = ob[8] / 1.0
        ingress = ob[10] / avg_time_interval * 8.0 / 1000000.0
        # forecast = np.asanyarray(ob[15],float)
        new_ob = np.array([queue_delay - prev_q, 50.0 * (prev_prob - prev_prev_p), throughput_dequeue - prev_th])
        prev_q = queue_delay
        prev_prev_p = prev_prob
        prev_th = throughput_dequeue
        # new_ob = np.array([queue_delay, prev_prob])


        ob_sliding_window = np.delete(ob_sliding_window, [0, 1, 2])
        ob_sliding_window = np.concatenate((ob_sliding_window, new_ob))

        # ********** Normalization

        # normalized_ob_wind[0::3] = np.maximum(np.minimum((ob_sliding_window[0::3] - 100.0)/100.0, 10.0), -10.0)
        # normalized_ob_wind[1::3] = np.maximum(np.minimum((ob_sliding_window[1::3] - 0.15) / 0.15, 10.0), -10.0)
        # normalized_ob_wind[2::3] = np.maximum(np.minimum((ob_sliding_window[2::3] - 5.0) / 5.0, 10.0), -10.0)

        normalized_ob_wind[0::3] = np.maximum(np.minimum(ob_sliding_window[0::3], 10.0), -10.0)
        normalized_ob_wind[1::3] = np.maximum(np.minimum(ob_sliding_window[1::3], 10.0), -10.0)
        normalized_ob_wind[2::3] = np.maximum(np.minimum(ob_sliding_window[2::3], 10.0), -10.0)

        # new_s = normalized_ob_wind[ob_sliding_window_size - state_dim::]
        mov_avg_th = 0.1 * min(throughput_dequeue, 40.0) / 10.0 + 0.9 * mov_avg_th
        mov_avg_q = 0.1 * queue_delay / 100.0 + 0.9 * mov_avg_q

        th_hist = np.delete(th_hist, [0])
        th_hist = np.concatenate((th_hist, [min(throughput_dequeue, 40)]))
        q_hist = np.delete(q_hist, [0])
        q_hist = np.concatenate((q_hist, [queue_delay]))
        p_hist = np.delete(p_hist, [0])
        p_hist = np.concatenate((p_hist, [prev_prob]))
        mean_p = np.average(p_hist) * 10.0
        dev_p = np.std(p_hist) * 10.0
        # th_hist = np.delete(th_hist,[0])
        # th_hist = np.concatenate((th_hist,[min(throughput_dequeue,40)]))
        # q_hist = np.delete(q_hist,[0])
        # q_hist = np.concatenate((q_hist,[queue_delay]))
        # p_hist = np.delete(p_hist,[0])
        # p_hist = np.concatenate((p_hist,[prev_prob]))
        # mean_p = np.average(p_hist) * 10.0
        # dev_p = np.std(p_hist) * 10.0


        q_hist_sorted = np.sort(q_hist)

        # new_s = np.concatenate([normalized_ob_wind[ob_sliding_window_size - (state_dim-3)::], mov_avg_drop,mov_avg_th,mov_avg_q])
        new_s = np.concatenate([np.reshape(normalized_ob_wind[ob_sliding_window_size - (state_dim - 8)::], (-1, 1)),
                                np.asarray(mov_avg_drop).reshape(-1, 1), np.asarray(mov_avg_th).reshape(-1, 1),
                                np.asarray(mov_avg_q).reshape(-1, 1),
                                np.asarray(q_hist_sorted[4] / 100.0).reshape(-1, 1),
                                np.asarray(np.average(th_hist) / 10.0).reshape(-1, 1)
                                   , np.asarray(mean_p).reshape(-1, 1), np.asarray((dev_p).reshape(-1, 1)),
                                np.asarray((ingress - throughput_dequeue)).reshape(-1, 1)])
        # new_s = np.concatenate([np.reshape(normalized_ob_wind[ob_sliding_window_size - (state_dim-3)::],(-1,1)), np.asarray(mov_avg_drop).reshape(-1,1),np.asarray(mov_avg_th).reshape(-1,1),np.asarray(mov_avg_q).reshape(-1,1)])
        # new_s = np.concatenate([normalized_ob_wind[ob_sliding_window_size - state_dim + 8::], np.maximum(np.minimum((forecast - 5.0) / 5.0, 1.0), -1.0)])

        # sleep(100)


        # Run Actor
        actor_action = actor.predict_target(np.reshape(new_s, (1, -1)))
        actor_action = actor_action[0][0]
        # new_prob = min(max(actor_action + random.normalvariate(0.0,0.1),0.0),action_bound) # ??/////////////additive
        new_prob = min(max(actor_action, 0.0), action_bound)  # ??/////////////additive
        # new_prob = min(max(actor_action + random.normalvariate(0.0,0.0),0.0),action_bound) # ??/////////////additive
        # new_prob = actor_action + prev_prob  # ??/////////////additive
        # new_prob = min(max( actor_action + prev_prob,0.0),1.0)  # ??/////////////additive
        # new_prob = actor_action[0][0]



        # if time() - channel_time < 50.0:
        # 	new_prob = random.uniform(0,0.4)
        # scale_prob = action_minmax(new_prob,0,action_bound) #* MinimumExecutorUpdateInterval/avg_time_interval
        # scale_prob = min(scale_prob,action_bound)
        # if new_prob < 0:
        #	scale_prob = 0.01
        # if new_prob > (action_bound + 0.1):
        #	scale_prob = action_bound / 2
        # Apply Drop Probability
        # mov_avg_drop = 0.8 * scale_prob + 0.2 * mov_avg_drop #change back prev_prob too!!
        # intf.SetDropRate(mov_avg_drop)

        # Generate Reward
        # prob_r = 1

        # if time() - channel_time < 300.0:
        # if new_prob < action_bound * 0.01:
        # prob_r = new_prob / (action_bound * 0.01)
        # prob_r = 0
        # if new_prob > action_bound * 0.9:
        # prob_r = 0
        # 	prob_r = (action_bound - new_prob)/(action_bound * 0.1)
        # if new_prob < 0 or new_prob > action_bound:
        # 	prob_r = 0



        # print throughput_dequeue, " ,", bw
        # throughput_list = ob_sliding_window[ob_sliding_window_size - state_dim+2::3] # careful here about when new states!
        # qdelay_list = ob_sliding_window[ob_sliding_window_size - state_dim+0::3]
        #
        # throughput_avg = min(np.average(throughput_list),40.0)
        # q_avg = np.average(qdelay_list)

        # if time() - lockertime > 0.1:
        # 	lockflag = False
        #
        # if (queue_delay > 300.0 or throughput_avg < 0.01) and lockflag == False:
        # 	lockertime = time()
        # 	lockflag = True
        # 	new_prob = 1
        #

        # if queue_delay > 0 and throughput_avg < 0.1 and outage_flag == False:
        # 	outage_flag = True
        # 	outage_start = time()
        # if outage_flag == True and time() - outage_start > 0.1:
        # 	outage_flag = False
        # 	new_prob = action_bound
        # if outage_flag:
        # 	new_prob = action_bound

        # Exploration:
        # exp_rate = max(EXPLORATION_RATE * (100.0 - (time()-channel_time))/100.0,0)
        # if random.uniform(0.0, 1.0) < exp_rate:
        # 	new_prob = random.uniform(0.0,0.6)

        # if new_prob == 1.0 and outage_flag == False:
        # 	outage_flag = True
        # 	outage_start = time()
        #
        # if time() - outage_start > 0.1 and outage_flag == True:
        # 	outage_flag = False
        # 	new_prob = 0

        # new_prob = 0.9 * new_prob + 0.1 * prev_prob
        # new_prob = min(0.9,new_prob)
        # if time() - channel_time < 20 and time()>channel_time and cnt < 4:
        # 	new_prob = random.uniform(0.0, 0.9/(cnt+1.0))
        # if time() - channel_time > 20:
        # 	channel_time += 20
        # 	cnt += 1



        # print "begin"
        # print throughput_dequeue
        # print forecast*1504*8/20
        #
        # new_prob = 0.0
        intf.SetDropRate(new_prob)

        # qdelay_list_sorted = np.sort(qdelay_list)
        # qdelay_95th = qdelay_list_sorted[14]
        # qdelay_mov_avg = 0.99 * queue_delay + 0.01 * qdelay_mov_avg
        # thrput_mov_avg = 0.99 * throughput_dequeue + 0.01 * thrput_mov_avg


        # ***************************************************************************
        # Control Delay to 50ms
        # reward = 0.5 * prob_r - 0.02 * np.abs(queue_delay - 50)
        # reward = - 0.02 * np.abs(queue_delay - 20)

        # Throughput Over Delay (Power)
        # reward = - 0.2 * (qdelay_95th / (throughput_avg + 0.1))


        # reward = - 0.02 * qdelay_mov_avg / (thrput_mov_avg + 0.1)

        # BE CAREFUL ABOUT THE 0.02

        # reward = 0.1 * throughput_dequeue/(queue_delay + 0.1)
        # reward /= bw
        # reward = ((throughput_dequeue / bw)**2 - queue_delay**2)/200.0
        # if new_prob < 0.0:
        #	reward = reward * 10.0 ** (100.0 * new_prob)
        # if new_prob > (action_bound/2.0):
        #	reward = reward * 10 ** (- new_prob + action_bound)
        # # Delay plus 1/throughput
        # *****************************************************************************
        # if ob[12]==0 or ob[13]==10000 or ob[8]>ob[12] or ob[8] < ob[13]:
        # 	print ob[12],ob[13],ob[8]
        # print ob
        # reward = np.average(th_hist) / (q_hist_sorted[4] + 40.0) * 3.0
        # reward = np.average(th_hist) / (q_hist_sorted[950] + 40.0) * 9.0
        # reward = throughput_dequeue / (queue_delay+40.0)
        # reward = min(max(np.average(th_hist)/5.0 - np.average(q_hist)/30.0,-0.9),0.9)
        # reward = -min(0.02 * np.abs(queue_delay-20.0),.99)
        if ob[0] > 0:
            reward = - (ob[11] / 25.0 + min(ob[13] / 200.0, 0.99)) / 2.0
        else:
            reward = - ob[11] / 25.0
        # reward = min(throughput_dequeue, 40.0) / (queue_delay + 40) * 5.0







        # reward = (min((throughput_dequeue/bw) ** 2, 1.0) + (0.1/(queue_delay+ 0.1)) ** 2) / 10.0
        # reward = min(throughput_dequeue / (queue_delay + 40) * 5.0, 0.99)
        # reward = 0.0
        # if throughput_dequeue > 3.0:
        # 	reward += 0.5
        # if queue_delay < 60:
        # 	reward += 0.5
        # reward = min(throughput_avg / (max(qdelay_list) + 40) * 5.0, 0.99)   ####THE BEST>>>>>>>>>>>
        # reward = min(min(60.0/(queue_delay+0.1), throughput_dequeue/3.0)/2.0,0.99)
        # ******************gaurbage
        # reward = throughput_avg / (qdelay_95th + 0.001) ** 2 /10000.0
        # reward = min(throughput_dequeue,1.0/(queue_delay+0.01))
        # reward = (0.1/(0.1 + scale_prob) + 0.02/(0.01+qdelay_95th) + 2.0 * throughput_avg) / 5.0
        # if new_prob <0.0:
        # 	reward = new_prob
        # # if new_prob > 0.9:
        # # 	reward = 0.9 - new_prob
        #
        # if mov_avg_drop > 0.38 and lockflag == False:
        # 	lockertime = time()
        # 	lockflag = True
        # # #
        # # # # if new_prob < -1.0 or new_prob > 2.0:
        # # # # 	new_prob = random.uniform(0.0,0.2)
        # if mov_avg_drop < 0.01 and lockflag ==False:
        # 	lockertime = time()
        # 	lockflag = True
        # # 	new_prob = random.uniform(0.3,0.4)
        # if lockflag == True and time() - lockertime > 20.0:
        #  	if outage_flag == False:
        # 		outage_flag = True
        # 		channel_time = time()
        # 	if time() - channel_time < 10.0 and outage_flag:
        #  		new_prob = random.uniform(0.05,0.25)


        #
        # else:
        # 	if lockflag == True:
        # 		print "lock released"
        # 		lockflag = False

        # # 	reward = reward - new_prob



        mov_avg_drop = 0.1 * max(new_prob, 0.0) + 0.9 * mov_avg_drop

        # TODO


        # Generate Experience
        experience = {'state': np.reshape(prev_state, (state_dim,)),
                      'action': np.reshape(prev_actor_action, (action_dim,)),
                      'reward': reward,
                      'next_state': np.reshape(new_s, (state_dim,)),
                      'queue_delay': queue_delay,
                      'throughput': throughput_dequeue,
                      'dropprob': new_prob}

        prev_prob = new_prob
        prev_actor_action = actor_action
        prev_state = new_s

        # Publish New Experience
        redis_server.publish('Experience', pickle.dumps(experience))

        # Check New Actor NN Variables
        # TODO Pickle
        actor_nn_params = None
        while True:
            msg = redis_pubsub.get_message()
            if msg:
                if msg['type'] == 'message':
                    actor_nn_params = pickle.loads(msg['data'])
            else:
                break

        # Update Actor NN Variables
        if actor_nn_params:
            actor_update_counter += 1
            # print('update')
            ts = time()
            for i in range(len(actor_nn_params)):
                # op = actor.network_params[i].assign(actor_nn_params[i])
                # tf_sess.run(op)
                actor.assign_params(i, actor_nn_params[i])

                # print('Time spent for update ',time() - ts)

        # Dump Infomation
        if time() - last_dump_ts > dump_time_interval:
            last_dump_ts = time()
            print(
            'EXECUTOR ', ep, ' : Last Drop Prob: ', new_prob, ' Last QDelay: ', queue_delay, ' Actor Update Counter: ',
            actor_update_counter, ' Avg Time Interval: ', avg_time_interval)
            actor_update_counter = 0
            ep += 1

        # Time Management
        time_passed = time() - start_time
        if time_passed < MinimumExecutorUpdateInterval:
            sleep(MinimumExecutorUpdateInterval - time_passed)

        time_passed = time() - start_time
        avg_time_interval = avg_time_interval * 0.75 + time_passed * 0.25


if __name__ == '__main__':
    main()
