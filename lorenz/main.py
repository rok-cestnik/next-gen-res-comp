import copy
from matplotlib import pyplot
from math import sqrt, tanh, cosh
import numpy as np
import random
from system_equations import *
from utils import *
from res_utils import *
from reservoir import *

# parameters
snoise = 0.01 # how much stabilizing measurement noise to add to inputs
NOV = 1 # number of variables
include_inputs = False # whether to include the inputs into the state
history = int(25/NOV) # how many delay values are taken
system = 'lorenz' # which system (roessler/lorenz/roessler_input)
# integration parameters
train_points = 100000 # how many points of the training signal are generated
val_points = 10000 # validation points
integration_time = 50000 # how long is the reconstruction integration
plot_time = 500 # how much of the reconstructed integrated signal is plotted (bounded by the integration_time)

def rescale(signal):
	mini = min(signal)
	maxi = max(signal)
	for i in range(len(signal)):
		signal[i] = ((signal[i]-mini)/(maxi-mini))*0.8+0.1

def rescale_back(val, mi, ma):
	return (val-0.1)/0.8*(ma-mi)+mi

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
if(system == 'roessler_input'):
	sampling = 15
	dt=0.01
if(system == 'lorenz'):
	sampling = 8
	dt=0.005
ders = get_ders(system)
# training signal
signal = generate_signal(ders, train_points, sampling, number_of_variables=NOV, include_inputs=include_inputs, warmup_time=3000.0, eps=0.5, tau=0.5, dt=dt)
x_min = min(signal[0])
x_max = max(signal[0])
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
print("filling A matrix")
for i in range(L-history):
	if(i%500 == 0):
		print("\t", i, "/", L-history)
	A.append(reservoir(seq, inputs[i]))
# the same for validation matrix
Av = []
print("filling A validation matrix")
for i in range(L_v-history):
	if(i%500 == 0):
		print("\t", i, "/", L_v-history)
	Av.append(reservoir(seq, inputs_v[i]))

pyplot.rcParams.update({"text.usetex": True, "font.size": 14, "font.family": "serif"})

# quick histogram plot
pyplot.hist([inp for sublist in A[0:1000] for inp in sublist], bins=500, edgecolor='black')
pyplot.xlabel(r"$p$")
pyplot.ylabel(r"Frequency($p$)")
pyplot.savefig("pics/"+system+"_NOV"+str(NOV)+"_inputs"+str(include_inputs)+"_history"+str(history)+"_histogram.png",dpi=300, bbox_inches='tight')
pyplot.show()

# linear algebra
print("linear algebra")
A = np.matrix(A)
b = [outputs[i] for i in range(len(outputs))]
AT = A.transpose()
ATA = AT*A
ATAI = np.linalg.inv(ATA)
ATAIAT = ATAI*AT
w = ATAIAT*b

# save weights
np.savetxt("saves/"+system+"_NOV"+str(NOV)+"_inputs"+str(include_inputs)+"_history"+str(history)+"_weights.txt", w, fmt='%lf')

# one step prediction plot
b_rec = A*w
pyplot.plot(b)
pyplot.plot(b_rec)
pyplot.show()
# validation
b_v = [outputs_v[i] for i in range(len(outputs_v))]
b_rec_v = Av*w
pyplot.plot(b_v)
pyplot.plot(b_rec_v)
pyplot.show()

# errors
ErT = error(b_rec,b)
ErV = error(b_rec_v,b_v)


# integration
print("integration")
signal_rec = res_integrate_signal(seq, w, inputs[0], integration_time)
signal_true = [inputs[i][-1] for i in range(integration_time)]

# re-scale it back
signal_rec = [[rescale_back(sr[0],x_min,x_max)] for sr in signal_rec]
signal_true = [rescale_back(signal[0][i],x_min,x_max) for i in range(len(signal[0]))]

# plot
#pyplot.plot(signal_true[0:plot_time])
#pyplot.plot(signal[0][history-1:history-1+plot_time])
pyplot.plot(signal_true[history-1:history-1+plot_time])
pyplot.plot(signal_rec[0:plot_time])
pyplot.legend(["true ODEs","reservoir"])
pyplot.xlabel(r"$t$")
pyplot.ylabel(r"$x$")
pyplot.savefig("pics/"+system+"_NOV"+str(NOV)+"_inputs"+str(include_inputs)+"_history"+str(history)+"_comparison.png", dpi=300, bbox_inches='tight')
pyplot.show()

if(system == 'roessler'):
	delay = 8
	plotshift = 0.9
if(system == 'roessler_input'):
	delay = 8
	plotshift = 0.9
if(system == 'lorenz'):
	delay = 3
	plotshift = 0.7

#pyplot.plot(signal[0][delay:integration_time], signal[0][:integration_time-delay], lw=0.05)
pyplot.plot(signal_true[delay:integration_time], signal_true[:integration_time-delay], lw=0.05)
pyplot.xlabel(r"$x(t)$")
pyplot.ylabel(r"$x(t-t_D)$")
pyplot.gca().set_aspect(1.5, adjustable='box')
pyplot.xticks([-20,-10,0,10,20])
pyplot.yticks([-20,-10,0,10,20])
pyplot.xlim([-20,20])
pyplot.ylim([-20,20])
pyplot.savefig("pics/"+system+"_NOV"+str(NOV)+"_inputs"+str(include_inputs)+"_history"+str(history)+"_attractor_true.png", dpi=300, bbox_inches='tight')
pyplot.show()

pyplot.plot(np.array(signal_rec[delay:integration_time]),signal_rec[:integration_time-delay], lw=0.05)
pyplot.xlabel(r"$x(t$)")
pyplot.ylabel(r"$x(t-t_D)$")
pyplot.gca().set_aspect(1.5, adjustable='box')
pyplot.xticks([-20,-10,0,10,20])
pyplot.yticks([-20,-10,0,10,20])
pyplot.xlim([-20,20])
pyplot.ylim([-20,20])
pyplot.savefig("pics/"+system+"_NOV"+str(NOV)+"_inputs"+str(include_inputs)+"_history"+str(history)+"_attractor_rec.png", dpi=300, bbox_inches='tight')
pyplot.show()
