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
NOV = 3 # number of variables
include_inputs = True # whether to include the inputs into the state
history = 25 # how many delay values are taken
system = 'roessler_input' # which system (roessler/lorenz/roessler_input)
# integration parameters
train_points = 100000 # how many points of the training signal are generated
val_points = 10000 # validation points
integration_time = 5000 # how long is the reconstruction integration
plot_time = 500 # how much of the reconstructed integrated signal is plotted (bounded by the integration_time)

def rescale(signal, mini, maxi):
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
if(system == 'roessler_input'):
	sampling = 15
	dt=0.01
if(system == 'lorenz'):
	sampling = 8
	dt=0.005
ders = get_ders(system)
# training signals
signal = generate_signal(ders, train_points, sampling, number_of_variables=NOV, include_inputs=include_inputs, warmup_time=3000.0, eps=0.5, tau=0.5, dt=dt)
x_min = min(signal[0])
x_max = max(signal[0])
y_min = min(signal[1])
y_max = max(signal[1])
z_min = min(signal[2])
z_max = max(signal[2])
n_min = -0.2+min(signal[3])
n_max = 0.1+max(signal[3])
rescale(signal[0], x_min, x_max)
rescale(signal[1], y_min, y_max)
rescale(signal[2], z_min, z_max)
rescale(signal[3], n_min, n_max)
L = len(signal[0])
flat_signal = flatten(signal)

pyplot.rcParams.update({"text.usetex": True, "font.size": 14, "font.family": "serif"})

# signal plot
start,stop = 1600,2000
times = [sampling*dt*i for i in range(start,stop)]
pyplot.plot(times, signal[0][start:stop])
pyplot.plot(times, signal[1][start:stop])
pyplot.plot(times, signal[2][start:stop])
pyplot.plot(times, [signal[3][i]-0.66 for i in range(start,stop)])
pyplot.yticks([0,0.2,0.4,0.6,0.8,1.0])
pyplot.xlabel(r"$t$")
pyplot.ylabel(r"$x,y,z,\eta$")
pyplot.savefig("pics/"+system+"_NOV"+str(NOV)+"_inputs"+str(include_inputs)+"_history"+str(history)+"_signal.png", dpi=300, bbox_inches='tight')
pyplot.show()

ENOV = len(signal) # effective number of variables
# inputs outputs
inputs = [[flat_signal[ENOV*i+j]+snoise*random.gauss(0,1) for j in range(ENOV*history)] for i in range(L-history)]
outputs = [[signal[n][i+history] for n in range(ENOV)] for i in range(L-history)]

# initialize the reservoir mixing sequence
seq = generate_pairs_sequence(inputs[0])

# fill the A matrix
A = []
print("filling A matrix")
L_full = len(inputs)
for i in range(L_full):
	if(i%500 == 0):
		print("\t", i, "/", L_full)
	A.append(reservoir(seq, inputs[i]))
	
# quick histogram plot
pyplot.hist([inp for sublist in A[0:1000] for inp in sublist], bins=500, edgecolor='black')
pyplot.xlabel("p")
pyplot.ylabel("Frequency")
pyplot.savefig("pics/"+system+"_NOV"+str(NOV)+"_inputs"+str(include_inputs)+"_history"+str(history)+"_histogram.png",dpi=300)
pyplot.show()

# linear algebra
print("linear algebra")
A = np.matrix(A)
b = [outputs[i] for i in range(len(outputs))]
w, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
w = np.matrix(w)
#AT = A.transpose()
#ATA = AT*A
#ATAI = np.linalg.inv(ATA)
#ATAIAT = ATAI*AT
#w = ATAIAT*b

# save weights
np.savetxt("saves/"+system+"_NOV"+str(NOV)+"_inputs"+str(include_inputs)+"_history"+str(history)+"_weights.txt", w, fmt='%lf')

# one step prediction plot
b_rec = A*w
pyplot.plot(b)
pyplot.plot(b_rec)
pyplot.show()

# errors
ErT = error(b_rec,b)

# bifurcation diagram
print("estimating the bifurcation diagram")
bd = bifurcation_diagram(ders, p_range=[-2.5,2.2], p_res=0.01)
bd_res = res_bifurcation_diagram(seq, w, inputs[0], p_range=[0.2,0.8], p_res=0.005, measure_steps=1000)
# re-scale bifurcation diagram
bd_res_scaled = [[n_min+(bd_res[0][i]-0.1)/0.8*(n_max-n_min) for i in range(len(bd_res[0]))], [x_min+(bd_res[1][i]-0.1)/0.8*(x_max-x_min) for i in range(len(bd_res[1]))]]
# plot
pyplot.scatter(bd[0], bd[1], s=0.2)
pyplot.scatter(bd_res_scaled[0], bd_res_scaled[1], s=1)
pyplot.plot([-2+4/100*i for i in range(100)],[0 for i in range(100)], c='#999999')
pyplot.plot([-2+4/100*i for i in range(100)],[4*exp(-2*(-2+4/100*i)**2) for i in range(100)], c='g')
pyplot.ylim([-1,12])
pyplot.xlabel(r"$\eta$")
pyplot.ylabel(r"max$_t(x)$")
pyplot.legend(["true ODEs", "reservoir"], scatterpoints=1, markerscale=6)
pyplot.savefig("pics/"+system+"_NOV"+str(NOV)+"_inputs"+str(include_inputs)+"_history"+str(history)+"_bifurcation.png", dpi=300, bbox_inches='tight')
pyplot.show()

