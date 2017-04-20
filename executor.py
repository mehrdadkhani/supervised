from rlcommon import *
from mahimahiInterface import RLMahimahiInterface
from ddpg import *


def action_minmax(a, lower, upper):
	ret = a
	if ret > upper :
		ret = upper
	if ret < lower:
		ret = lower

	return ret

def main():
	#Initialize Redis for IPC
	redis_server = redis.StrictRedis(host='localhost', port=6379, db=0)
	redis_pubsub = redis_server.pubsub()
	redis_pubsub.subscribe('Actor')

	#Initialize Environment Interface
	intf = RLMahimahiInterface()
	arg = str(sys.argv)
	# intf.ConnectToMahimahi(ip = sys.argv[1])
	intf.ConnectToMahimahi(ip = "100.64.0.4")
	intf.SetRLState(IsRLEnabled)

	avg_time_interval = SamplingTime
	ob_sliding_window_size = 6
	ob_sliding_window = np.zeros(ob_sliding_window_size,dtype = float)
	last_dump_ts = time()
	ep = 0
	last_probe_time = 0

	while True:
		#Start Time
		start_time = time()
		
		# Get Observations from Environment
		ob = intf.GetState(dump=False)

		# For modeling the TCP throughput, we should look at the up-link queue ingress
		th_t = ob[10] * 8 / (time() - last_probe_time) / 1000000.0		#Enqueue rate (TCP client sits on the enqueue side)
		last_probe_time = time()		#To find the time interval
		p_t = random.uniform(0, 0.3)
		intf.SetDropRate(p_t)
		new_ob = np.array([th_t, p_t])

		ob_sliding_window = np.delete(ob_sliding_window, [0, 1])
		ob_sliding_window = np.concatenate((ob_sliding_window, new_ob))
		output = np.zeros((out_dim,), int)
		output[min(out_dim - 1, int(th_t/MAX_THRPUT * out_dim) - 1)] = 1 		# Making the output one-hot

		# Generate Experience
		experience = {'input': np.reshape(prev_state, (in_dim, )), 'output': output}
		prev_state = ob_sliding_window

		#Publish New Experience
		redis_server.publish('Experience',pickle.dumps(experience))



		#Check New Actor NN Variables
		#TODO Pickle
		# actor_nn_params = None
		# while True:
		# 	msg = redis_pubsub.get_message()
		# 	if msg:
		# 		if msg['type'] == 'message':
		# 			actor_nn_params = pickle.loads(msg['data'])
		# 	else:
		# 		break



		# #Update Actor NN Variables
		# if actor_nn_params:
		# 	actor_update_counter += 1
		# 	#print('update')
		# 	ts = time()
		# 	for i in range(len(actor_nn_params)):
		# 		#op = actor.network_params[i].assign(actor_nn_params[i])
		# 		#tf_sess.run(op)
		# 		NN.assign_params(i, actor_nn_params[i])
        #
		# 	#print('Time spent for update ',time() - ts)




		# Dump Infomation
		if time() - last_dump_ts > dump_time_interval:
			last_dump_ts = time()
			print('EXECUTOR ',ep,' : State: ',p_t,'thrput: ',th_t, 'Output:',output, 'avg_time:',avg_time_interval)
			actor_update_counter = 0
			ep += 1


		# Time Management
		time_passed = time() - start_time
		if time_passed < SamplingTime:
			sleep(SamplingTime - time_passed)


		time_passed = time() - start_time
		avg_time_interval = avg_time_interval * 0.75 + time_passed * 0.25



if __name__ == '__main__':
	main()
