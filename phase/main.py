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
include_inputs = False # whether to include the inputs into the state
history = int(25/NOV) # how many delay values are taken
system = 'roessler_input' # which system (roessler/lorenz/roessler_input)
# integration parameters
train_points = 100000 # how many points of the training signal are generated
val_points = 1000 # validation points
integration_time = 5000 # how long is the reconstruction integration
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
y_min = min(signal[1])
y_max = max(signal[1])
z_min = min(signal[2])
z_max = max(signal[2])
for n in range(len(signal)):
	rescale(signal[n])
L = len(signal[0])
flat_signal = flatten(signal)
# validation signal
signal_v = generate_signal(ders, val_points, sampling, number_of_variables=NOV, include_inputs=include_inputs, eps=0.5, tau=0.5, dt=dt)
xv_min = min(signal_v[0])
xv_max = max(signal_v[0])
yv_min = min(signal_v[1])
yv_max = max(signal_v[1])
zv_min = min(signal_v[2])
zv_max = max(signal_v[2])
for n in range(len(signal_v)):
	rescale(signal_v[n])
L_v = len(signal_v[0])
flat_signal_v = flatten(signal_v)

ENOV = len(signal) # effective number of variables
# inputs outputs
inputs = [[flat_signal[ENOV*i+j]+snoise*random.gauss(0,1) for j in range(ENOV*history)] for i in range(L-history)]
outputs = [[signal[n][i+history] for n in range(ENOV)] for i in range(L-history)]
# the same for validation
inputs_v = [[flat_signal_v[ENOV*i+j] for j in range(ENOV*history)] for i in range(L_v-history)]
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

# reservoir phases
Tres = res_estimate_period(seq, w, inputs[0])
res_phases = []
phase_x = []
phase_y = []
phase_z = []
print("measuring reservoir phases...")
for t in range(0,len(inputs_v),2):
	print("\tt = "+str(int(t/2))+"/"+str(int(len(inputs_v)/2)))
	phase_x.append(inputs_v[t][-3])
	phase_y.append(inputs_v[t][-2])
	phase_z.append(inputs_v[t][-1])
	res_phases.append(res_phase(seq, w, inputs_v[t], period=Tres))

# true phases
T = oscillator_period(ders)
true_phases = []
print("measuring true phases...")
for t in range(len(phase_x)):
	print("\tt = "+str(t)+"/"+str(len(phase_x)))
	state = [rescale_back(phase_x[t],x_min,x_max), rescale_back(phase_y[t],y_min,y_max), rescale_back(phase_z[t],z_min,z_max)]
	true_phases.append(oscillator_phase(ders,state,period=T))

# interpolated phases
int_res_phases = []
int_true_phases = []
for t in range(int(len(inputs_v)/2)-1):
	int_res_phases.append(res_phases[t])
	int_true_phases.append(true_phases[t])
	# compute mid-point using complex interpolation
	z1 = np.exp(1j * 2*pi * res_phases[t])
	z2 = np.exp(1j * 2*pi * res_phases[t+1])
	mid_z = (z1 + z2) / 2
	mid_phase = (np.angle(mid_z)%(2*pi))/(2*pi)
	int_res_phases.append(mid_phase)
	z1 = np.exp(1j * 2*pi * true_phases[t])
	z2 = np.exp(1j * 2*pi * true_phases[t+1])
	mid_z = (z1 + z2) / 2
	mid_phase = (np.angle(mid_z)%(2*pi))/(2*pi)
	int_true_phases.append(mid_phase)
int_phase_x = [inputs_v[t][-3] for t in range(len(int_res_phases))]
int_phase_y = [inputs_v[t][-2] for t in range(len(int_res_phases))]
# x,y corresponding to interpolated
int_phase_x = []
int_phase_y = []
for t in range(0,len(inputs_v)):
	int_phase_x.append(rescale_back(inputs_v[t][-3], x_min, x_max))
	int_phase_y.append(rescale_back(inputs_v[t][-2], y_min, y_max))

#pyplot.plot(int_phase_x, int_res_phases)
#pyplot.plot(int_phase_x, int_true_phases)
#pyplot.show()

pyplot.scatter(true_phases, res_phases)
pyplot.show()

#pyplot.scatter(phase_x, phase_y, c=true_phases, cmap='hsv')
#pyplot.show()

pyplot.rcParams.update({"text.usetex": True, "font.size": 14, "font.family": "serif"})

def phase_plot(xs,ys,ps,name):
	from matplotlib.collections import LineCollection
	# create segments
	points = np.array([xs, ys]).T.reshape(-1, 1, 2)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)
	# create a LineCollection
	lc = LineCollection(segments, cmap='hsv')
	lc.set_array(ps)
	lc.set_linewidth(2)
	# figure
	fig, ax = pyplot.subplots()
	line = ax.add_collection(lc)
	ax.autoscale()
	ax.set_aspect(1.5)#'equal')
	#pyplot.colorbar(line, ax=ax, label='Phase')
	pyplot.xlabel(r'$x$')
	pyplot.ylabel(r'$y$')
	pyplot.savefig("pics/phase_trajectory_"+name+".png", dpi=300, bbox_inches='tight')
	pyplot.show()

phase_plot(int_phase_x,int_phase_y,int_true_phases,"true")
phase_plot(int_phase_x,int_phase_y,int_res_phases,"res")
