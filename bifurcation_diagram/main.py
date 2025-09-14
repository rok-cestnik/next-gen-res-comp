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
train_points = 25000 # how many points of the training signal are generated
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
signal1 = generate_signal(ders, train_points, sampling, number_of_variables=NOV, include_inputs=include_inputs, warmup_time=3000.0, constant_input=-0.5, eps=0.5, tau=0.5, dt=dt)
signal2 = generate_signal(ders, train_points, sampling, number_of_variables=NOV, include_inputs=include_inputs, warmup_time=3000.0, constant_input=0.0, eps=0.5, tau=0.5, dt=dt)
signal3 = generate_signal(ders, train_points, sampling, number_of_variables=NOV, include_inputs=include_inputs, warmup_time=3000.0, constant_input=0.5, eps=0.5, tau=0.5, dt=dt)
signal4 = generate_signal(ders, train_points, sampling, number_of_variables=NOV, include_inputs=include_inputs, warmup_time=3000.0, constant_input=1.0, eps=0.5, tau=0.5, dt=dt)
#signal5 = generate_signal(ders, train_points, sampling, number_of_variables=NOV, include_inputs=include_inputs, warmup_time=3000.0, constant_input=2.5, eps=0.5, tau=0.5, dt=dt)
x_min = min(map(min, [signal1[0], signal2[0], signal3[0], signal4[0]]))#, signal5[0]]))
x_max = max(map(max, [signal1[0], signal2[0], signal3[0], signal4[0]]))#, signal5[0]]))
y_min = min(map(min, [signal1[1], signal2[1], signal3[1], signal4[1]]))#, signal5[1]]))
y_max = max(map(max, [signal1[1], signal2[1], signal3[1], signal4[1]]))#, signal5[1]]))
z_min = min(map(min, [signal1[2], signal2[2], signal3[2], signal4[2]]))#, signal5[2]]))
z_max = max(map(max, [signal1[2], signal2[2], signal3[2], signal4[2]]))#, signal5[2]]))
n_min = -1+min(map(min, [signal1[3], signal2[3], signal3[3], signal4[3]]))#, signal5[3]]))
n_max = 0.7+max(map(max, [signal1[3], signal2[3], signal3[3], signal4[3]]))#, signal5[3]]))
rescale(signal1[0], x_min, x_max)
rescale(signal2[0], x_min, x_max)
rescale(signal3[0], x_min, x_max)
rescale(signal4[0], x_min, x_max)
#rescale(signal5[0], x_min, x_max)
rescale(signal1[1], y_min, y_max)
rescale(signal2[1], y_min, y_max)
rescale(signal3[1], y_min, y_max)
rescale(signal4[1], y_min, y_max)
#rescale(signal5[1], y_min, y_max)
rescale(signal1[2], z_min, z_max)
rescale(signal2[2], z_min, z_max)
rescale(signal3[2], z_min, z_max)
rescale(signal4[2], z_min, z_max)
#rescale(signal5[2], z_min, z_max)
rescale(signal1[3], n_min, n_max)
rescale(signal2[3], n_min, n_max)
rescale(signal3[3], n_min, n_max)
rescale(signal4[3], n_min, n_max)
#rescale(signal5[3], n_min, n_max)
L = len(signal1[0])
flat_signal1 = flatten(signal1)
flat_signal2 = flatten(signal2)
flat_signal3 = flatten(signal3)
flat_signal4 = flatten(signal4)
#flat_signal5 = flatten(signal5)

ENOV = len(signal1) # effective number of variables
# inputs outputs
inputs = [[flat_signal1[ENOV*i+j]+snoise*random.gauss(0,1) for j in range(ENOV*history)] for i in range(L-history)]+[[flat_signal2[ENOV*i+j]+snoise*random.gauss(0,1) for j in range(ENOV*history)] for i in range(L-history)]+[[flat_signal3[ENOV*i+j]+snoise*random.gauss(0,1) for j in range(ENOV*history)] for i in range(L-history)]+[[flat_signal4[ENOV*i+j]+snoise*random.gauss(0,1) for j in range(ENOV*history)] for i in range(L-history)]#+[[flat_signal5[ENOV*i+j]+snoise*random.gauss(0,1) for j in range(ENOV*history)] for i in range(L-history)]
outputs = [[signal1[n][i+history] for n in range(ENOV)] for i in range(L-history)]+[[signal2[n][i+history] for n in range(ENOV)] for i in range(L-history)]+[[signal3[n][i+history] for n in range(ENOV)] for i in range(L-history)]+[[signal4[n][i+history] for n in range(ENOV)] for i in range(L-history)]#+[[signal5[n][i+history] for n in range(ENOV)] for i in range(L-history)]

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

pyplot.rcParams.update({"text.usetex": True, "font.size": 14, "font.family": "serif"})

# bifurcation diagram
print("estimating the bifurcation diagram")
bd = bifurcation_diagram(ders, p_range=[-2.5,2.2], p_res=0.01)
bd_res = res_bifurcation_diagram(seq, w, inputs[0], p_range=[0.2,0.8], p_res=0.005, measure_steps=1000)
# re-scale bifurcation diagram
bd_res_scaled = [[n_min+(bd_res[0][i]-0.1)/0.8*(n_max-n_min) for i in range(len(bd_res[0]))], [x_min+(bd_res[1][i]-0.1)/0.8*(x_max-x_min) for i in range(len(bd_res[1]))]]
# plot
pyplot.scatter(bd[0], bd[1], s=0.2)
pyplot.scatter(bd_res_scaled[0], bd_res_scaled[1], s=1)
pyplot.plot([-0.5,-0.5],[0,12],c='g')
pyplot.plot([0.,0.],[0,12],c='g')
pyplot.plot([0.5,0.5],[0,12],c='g')
pyplot.plot([1.0,1.0],[0,12],c='g')
pyplot.xlabel(r"$\eta$")
pyplot.ylabel(r"max$_t(x)$")
pyplot.legend(["true ODEs", "reservoir"], scatterpoints=1, markerscale=6)
pyplot.ylim([0,12])
pyplot.savefig("pics/"+system+"_NOV"+str(NOV)+"_inputs"+str(include_inputs)+"_history"+str(history)+"_bifurcation.png", dpi=300, bbox_inches='tight')
pyplot.show()

