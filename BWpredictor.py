from ddpg import *
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import glob


# MAX_BW = 50.0
BW_AVG_INT = 20   #BW averaging interval in ms

# TRACES_DIR = './traces/*'
# trace_files = glob.glob(TRACES_DIR)
# for filename in trace_files:
#     x = np.loadtxt(filename, dtype=int)
#     intervals = np.array([0, 2, 5, 7, 10, 15, 20, x[-1]])     #ATTENTION: different traces may have different distributions

x = np.loadtxt('./traces/Verizon-LTE-driving.up',dtype=int)
intervals = np.array([0, 2, 5, 7, 10, 15, 20, x[-1]])

LEVELS = len(intervals) - 1
BATCH_SIZE = 100000
LEARNING_RATE = 0.05
# SUMMARY_DIR = './summary'
logs_path = './run1'

BW = np.histogram(x,bins = range(0,x[-1],BW_AVG_INT))
BW = BW[0]

dbw = np.zeros((len(BW),int(LEVELS)), float)


# intervals = [0,2,5,7,10,12,15,17,20,25,30,x[-1]]


for i in xrange(0,len(BW)):
    # print len(dbw[i])
    # print np.histogram(BW[i], bins=intervals)[0]
    dbw[i]= np.histogram(BW[i], bins=intervals)[0]   # Making the output one-hot
    # print dbw[i],BW[i]
    # dbw[i][min(int(LEVELS - 1.0), max(int(BW[i]/MAX_BW * LEVELS),0))]=1.0    # Making the output one-hot
#
# plt.hist(BW)
# plt.show()		# b1 = tf.Variable(tf.zeros([20]))


# result = np.correlate(BW, BW, mode='full')
# plt.plot(result[result.size/2:])
# plt.show()

tf_sess = tf.Session()
BW_NN = Network(tf_sess, 5, int(LEVELS), LEARNING_RATE)
tf_sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()

# summary_ops, summary_vars = build_summaries()
# writer = tf.train.SummaryWriter(SUMMARY_DIR, tf_sess.graph)
writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

time_stamp = time()
avg_loss = 0.0
step = 0
iteration = 0
avg_accuracy = 0.0

# batch_indx = np.random.choice(len(BW) - 10, size=BATCH_SIZE, replace=True)  #JUST FOR A TEST

print "begin"

while True:
    batch_indx = np.random.choice(len(BW)-10, size=BATCH_SIZE, replace=True)
    # batch_input = np.concatenate([dbw[batch_indx],dbw[batch_indx+1]],axis=1)/1.0
    # print batch_indx[20]
    # print BW[batch_indx[20]], BW[batch_indx[20]+1],BW[batch_indx[20]+2]
    batch_input = np.concatenate([BW[batch_indx].reshape(-1,1),BW[batch_indx+1].reshape(-1,1),
                                 BW[batch_indx + 2].reshape(-1, 1), BW[batch_indx+3].reshape(-1,1),
                                  BW[batch_indx+4].reshape(-1,1)], axis=1)/50.0
    batch_output = dbw[batch_indx+5]/1.0
    # print batch_input[20],batch_output[20]
    loss, _, summary = BW_NN.train(batch_input,batch_output)
    iteration += 1
    avg_loss = 0.25 * loss + 0.75 * avg_loss
    # avg_accuracy = 0.25 * accuracy + 0.75 * avg_accuracy


    if time() - time_stamp > 2.0:
        step += 1
        time_stamp = time()
        print 'step: ', step, 'iteration: ', iteration, 'avg_loss: ', avg_loss, 'avg_accuracy: ', avg_accuracy   #BW_NN.acc_eval(batch_input, batch_output)


        writer.add_summary(summary,step)

        # Saving the parameters
        if step % 100 == 0:
            saver.save(tf_sess,'NN_step', global_step = step)
            print 'parameters saved to NN_step-%(step)d' % {'step': step}






