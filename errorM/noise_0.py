import copy
from matplotlib import pyplot
from math import sqrt, tanh, cosh
import numpy as np
import random
from utils import *
from system_equations import *

ErTs = []
ErVs = []

for M in range(1,2000,1):
	print("M="+str(M))
	########### RESERVOIR ############
	def combine_two(v1,v2):
		return (1-v1)**v2
	def generate_pairs_sequence(inp):
		seq = []
		random.seed()
		L = len(inp)
		while len(seq) < M:
			candidate = [floor((L+len(seq))*random.random()), floor((L+len(seq))*random.random())]
			if candidate not in seq:
				seq.append(candidate)
		return seq
	def reservoir(seq, inp):
		L = len(inp)
		out = inp.copy()
		for r in range(M):
			out.append(combine_two(out[seq[r][0]],out[seq[r][1]]))
		return out[L:]
	########### RESERVOIR END ############
	
	# parameters
	history = 20 # how many delay values are taken
	snoise = 0.0 # how much stabilizing measurement noise to add to inputs
	NOV = 1 # number of variables
	include_inputs = False # whether to include the inputs into the state
	system = 'lorenz' # which system (roessler/lorenz/roessler_noise)
	# integration parameters
	train_points = 100000 # how many points of the training signal are generated
	val_points = 100000 # validation points

	def rescale(signal):
		mini = min(signal)
		maxi = max(signal)
		for i in range(len(signal)):
			signal[i] = ((signal[i]-mini)/(maxi-mini))*0.8+0.1

	def flatten(signal):
		flat_signal = []
		L = len(signal[0])
		for i in range(L):
			for n in range(len(signal)):
				flat_signal.append(signal[n][i])
		return flat_signal

	def error(pred,true):
		er = 0
		for i in range(len(pred)):
			er += (pred[i][0,0]-true[i][0])**2
		return er/len(pred)

	# signals
	if(system == 'roessler'):
		sampling = 15
		dt=0.01
	if(system == 'roessler_noise'):
		sampling = 15
		dt=0.01
	if(system == 'lorenz'):
		sampling = 8
		dt=0.005
	ders = get_ders(system)
	# training signal
	signal = generate_signal(ders, train_points, sampling, number_of_variables=NOV, include_inputs=include_inputs, warmup_time=3000.0, eps=0.5, tau=0.5, dt=dt)
	for n in range(len(signal)):
		rescale(signal[n])
	L = len(signal[0])
	flat_signal = flatten(signal)
	# validation signal
	signal_v = generate_signal(ders, val_points, sampling, number_of_variables=NOV, include_inputs=include_inputs, eps=0.5, tau=0.5, dt=dt)
	for n in range(len(signal)):
		rescale(signal_v[n])
	L_v = len(signal_v[0])
	flat_signal_v = flatten(signal_v)

	ENOV = len(signal) # effective number of variables
	# inputs outputs
	inputs = [[flat_signal[ENOV*i+j]+snoise*random.gauss(0,1) for j in range(ENOV*history)] for i in range(L-history)]
	outputs = [[signal[n][i+history] for n in range(ENOV)] for i in range(L-history)]
	# the same for validation
	inputs_v = [[flat_signal_v[ENOV*i+j]+snoise*random.gauss(0,1) for j in range(ENOV*history)] for i in range(L_v-history)]
	outputs_v = [[signal_v[n][i+history] for n in range(ENOV)] for i in range(L_v-history)]

	# initialize the reservoir mixing sequence
	seq = generate_pairs_sequence(inputs[0])

	# fill the A matrix
	A = []
	#print("filling A matrix")
	for i in range(L-history):
		#if(i%500 == 0):
		#	print("\t", i, "/", L-history)
		A.append(reservoir(seq, inputs[i]))
	# the same for validation matrix
	Av = []
	#print("filling A validation matrix")
	for i in range(L_v-history):
		#if(i%500 == 0):
		#	print("\t", i, "/", L_v-history)
		Av.append(reservoir(seq, inputs_v[i]))

	# linear algebra
	#print("linear algebra")
	A = np.matrix(A)
	b = [outputs[i] for i in range(len(outputs))]
	AT = A.transpose()
	ATA = AT*A
	ATAI = np.linalg.inv(ATA)
	ATAIAT = ATAI*AT
	w = ATAIAT*b
	
	# one step prediction plot
	b_rec = A*w
	b_v = [outputs_v[i] for i in range(len(outputs_v))]
	b_rec_v = Av*w
	
	# errors
	ErT = error(b_rec,b)
	ErV = error(b_rec_v,b_v)
	ErTs.append(ErT)
	ErVs.append(ErV)
	print("\tErT,ErV = "+str(ErT)+" "+str(ErV))
	with open("saves/"+system+"_errors_0.txt","a") as file:
		file.write(str(M)+", "+str(ErT)+", "+str(ErV)+"\n")

