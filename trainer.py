from rlcommon import *
from ddpg import *
from replay_buffer import ReplayBuffer


def main():
    np.set_printoptions(threshold=np.inf)
    # Initialize Redis for IPC
    redis_server = redis.StrictRedis(host='localhost', port=6379, db=0)
    redis_pubsub = redis_server.pubsub()
    redis_cmd = redis_server.pubsub()
    redis_pubsub.subscribe('Experience')
    redis_cmd.subscribe('cmd')

    # Initialize Monitor
    # mon = Popen('python3 parameter_monitor.py', stdin=PIPE, shell=True)

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    # Initialize Tensorflow (NN)

    tf_sess = tf.Session()
    NN = Network(tf_sess, in_dim, out_dim, LEARNING_RATE)


    summary_ops, summary_vars = build_summaries()

    tf_sess.run(tf.initialize_all_variables())
    writer = tf.train.SummaryWriter(SUMMARY_DIR, tf_sess.graph)

    # save neural net parameters
    # saver = tf.train.Saver()

    # nn_model = None
    # if nn_model is not None:  # nn_model is the path to file
    #     saver.restore(sess, nn_model)
    #     print("Model restored.")

    # # Initialize target network weights
    # actor.update_target_network()
    # critic.update_target_network()

    # Initialize Variables
    td_loss_sum = 0
    td_loss = 0
    iteration_counter = 0

    avg_time_interval = MinimumTrainerUpdateInterval

    last_dump_ts = time()
    time_stamp_rec = time()

    cmd = "echo start > super_ep.txt"
    Popen(cmd, shell=True).wait()

    ep = 0

    # IsTraining = True

    # saver.restore(tf_sess, 'jan22VII')
    # main loop of actor
    while True:
        # Start time
        start_time = time()

        # Update Replay Buffer
        HasNewExperience = False
        dumpstr = ''
        while True:
            msg = redis_pubsub.get_message()
            if msg:
                if msg['type'] == 'message':
                    HasNewExperience = True
                    new_experience = pickle.loads(msg['data'])
                    # Add new experience to replay buffer
                    replay_buffer.add(
                        np.reshape(new_experience['input'], (in_dim,)),
                        np.reshape(new_experience['output'], (out_dim,)),
                        0,
                        False,
                        0)




                    # dumpstr = "%(td_loss)7.2f %(qlen)f %(dp)f %(throughput)f %(reward)f " % {"td_loss": td_loss, "qlen": new_experience['queue_delay'], "dp": new_experience['dropprob'], "throughput": new_experience['throughput'], "reward": new_experience['reward']}
                    # cmd = "echo '%s' >> log.txt" % dumpstr
                    # Popen(cmd, shell=True).wait()


                    #mon.stdin.write('%f,%f,%f,%f\n' % (new_experience['queue_delay'], new_experience['dropprob'], new_experience['throughput'],new_experience['power']))
            else:
                break

        if HasNewExperience == False:
            sleep(0.01)  # 10ms
            continue

        # Update Critic and Actor
        if replay_buffer.size() > MINIBATCH_SIZE:
            iteration_counter += 1
            # s_batch1, a_batch1, r_batch1, t_batch1, s2_batch1 = \
            #     replay_buffer.sample_batch(MINIBATCH_SIZE-64)
            in_batch, out_batch, r_batch, t_batch, s2_batch = \
                replay_buffer.sample_batch(MINIBATCH_SIZE)



            NN.train(in_batch,out_batch)


        # Send new actor neural network to executor
        # TODO

        # nn_params = tf_sess.run(NN.network_params)
        # redis_server.publish('Actor', pickle.dumps(nn_params))

        # if time() - last_dump_ts > dump_time_interval:
        #     last_dump_ts = time()
        #     if iteration_counter == 0:
        #         iteration_counter = -1
        #     print(
        #     'TRAINER ', ep, ' : Iteration Counter: ', iteration_counter, ' TdLoss: ', td_loss_sum / iteration_counter,'rewards: ', new_experience['reward'])#, 'qbatch: ', np.average(s_batch[:,0::3]),'pbatch: ', np.average(s_batch[:,1::3]),'thbatch: ', np.average(s_batch[:,2::3]))
        #
        #     #me:
        #     # a_weights = actor.get_w()
        #     # print a_weights[7]
        #     # stract = ""
        #     # for i in xrange(len(actor_weights[3])):
        #     #     stract+= str(actor_weights[3][i]) + " "
        #     # cmd = "echo '%s' >> actor_w.txt" % stract
        #     # Popen(cmd, shell=True).wait()
        #
        #     ep += 1
        #     iteration_counter = 0
        #     td_loss_sum = 0

        # Handle Control Command

        while True:
            msg = redis_cmd.get_message()
            if msg:
                if msg['type'] == 'message':
                    cmd = pickle.loads(msg['data'])
                    if cmd['cmd'] == 'load':
                        print('Load Model: ', cmd['name'])
                        saver.restore(tf_sess, cmd['name'])
                        pass

                    if cmd['cmd'] == 'store':
                        print('Store Model: ', cmd['name'])
                        saver.save(tf_sess, cmd['name'])
                        pass

                    if cmd['cmd'] == 'stop_training':
                        print('Stop Training')
                        IsTraining = False
                        pass

                    if cmd['cmd'] == 'resume_training':
                        print('Resume Training')
                        IsTraining = True
                        pass
                    # if cmd['cmd'] == 'nomore':
                    #     print('NOMORE')
                    #     nomore_flag = True
                    #     pass
                    #
                    # if cmd['cmd'] == 'more':
                    #     print('MORE')
                    #     nomore_flag = False
                    #     pass
                    # if cmd['cmd'] == 'exp':
                    #     print('EXP:')
                    #     exploration_flag = not exploration_flag
                    #     print exploration_flag
                    #     pass
                    

            else:
                break

        # Time Management
        # time_passed = time() - start_time
        # if time_passed < MinimumTrainerUpdateInterval:
        #     sleep(MinimumTrainerUpdateInterval - time_passed)
        #
        # time_passed = time() - start_time
        # avg_time_interval = avg_time_interval * 0.75 + time_passed * 0.25


if __name__ == '__main__':
    main()








