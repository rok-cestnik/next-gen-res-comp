import numpy as np
import random
from math import pi, log, exp, sqrt, floor, sin, cos, atan

	
def one_step_integrator(state, ders, dt, inp=0):
	"""RK4 integrates state with derivative and input for one step of dt
	
	:param state: state of the variables
	:param ders: derivative functions
	:param dt: time step
	:param inp: input (default 0)
	:return: state after one integration step"""
	D = len(state)
	# 1
	k1 = [ders[i](state,inp=inp) for i in range(D)]
	# 2
	state2 = [state[i]+k1[i]*dt/2.0 for i in range(D)]
	k2 = [ders[i](state2,inp=inp) for i in range(D)]
	# 3
	state3 = [state[i]+k2[i]*dt/2.0 for i in range(D)] 
	k3 = [ders[i](state3,inp=inp) for i in range(D)]
	# 4
	state4 = [state[i]+k3[i]*dt for i in range(D)] 
	k4 = [ders[i](state4,inp=inp) for i in range(D)]
	# put togeather
	statef = [state[i] + (k1[i]+2*k2[i]+2*k3[i]+k4[i])/6.0*dt for i in range(D)]
	return statef


def generate_signal(ders, n, sampling, number_of_variables=1, include_inputs=False, warmup_time=1000.0, constant_input=False, tau=3.0, eps=0, dt=0.01):
	"""generates signal for the oscillator driven by correlated noise
	
	:param ders: a list of state variable derivatives
	:param n: length of time series
	:param sampling: the sampling rate
	:param number_of_variables: the number of variables returned, not including the input (default 1)
	:param inlcude_inputs: whether or not to include inputs (default False)
	:param warmup_time: the time for relaxing to the stationary regime (default 1000)
	:param constant_input: the constant input fed into the dynamics, if False then noisy input is used instead (default False)
	:param tau: noise correlation time (default 3.0)
	:param eps: noise strength (default 0)
	:param dt: time step (default 0.01)
	:returns: time series of the signal and driving noise"""
	# initial conditions
	state = [random.gauss(0,0.2) for i in range(len(ders))]
	resS = [[] for i in range(len(ders))]
	I = 0.0
	inputs = []
	# warmup
	for i in range(round(warmup_time/dt)):
		if constant_input is not False:
			I = constant_input
		else:
			if eps>0:
				I = I - (I/tau - eps*sqrt(2/tau)*random.gauss(0,1)/sqrt(dt))*dt
		state = one_step_integrator(state, ders, dt, inp=I)
	# real integration
	for i in range(n*sampling):
		if constant_input is not False:
			I = constant_input
		else:
			if eps>0:
				I = I - (I/tau - eps*sqrt(2/tau)*random.gauss(0,1)/sqrt(dt))*dt
		if include_inputs:
			inputs.append(I)
		state = one_step_integrator(state, ders, dt, inp=I)
		for c in range(len(ders)):
			resS[c].append(state[c])
	# return list
	rl = [resS[i][::sampling] for i in range(number_of_variables)]
	if include_inputs:
		rl = rl+[inputs[::sampling]]
	return rl


def bifurcation_diagram(ders, p_range=[0,3], p_res = 0.05, warmup_time=1000.0, small_warmup=250.0, measure_time=500.0, dt=0.01):
	"""gets the bifurcation diagram of extremal points
	
	:param ders: a list of state variable derivatives
	:param p_range: range of the control parameter (default 0-3)
	:param p_res: resolution of the control parameter (default 0.05)
	:param warmup_time: the time for relaxing to the stationary regime (default 1000)
	:param small_warmup: the time for relaxing after small parameter shift (default 250)
	:param measure_time: the time for measuring extremal points (default 500)
	:param dt: time step (default 0.01)
	:returns: time series of the signal and driving noise"""
	# initial conditions
	state = [random.gauss(0,0.2) for i in range(len(ders))]
	biffd_args = []
	biffd = []
	# warmup
	buff = [0,0,0] # 3 valued signal buffer
	for i in range(round(warmup_time/dt)):
		state = one_step_integrator(state, ders, dt, inp=p_range[0])
		buff[0]=buff[1]; buff[1]=buff[2]; buff[2]=state[0]
	# bifurcation diagram
	p_vals = [p_range[0]+p_res*i for i in range(round((p_range[1]-p_range[0])/p_res))]
	for p in p_vals:
		# small warmup
		for i in range(round(small_warmup/dt)):
			state = one_step_integrator(state, ders, dt, inp=p)
			buff[0]=buff[1]; buff[1]=buff[2]; buff[2]=state[0]
		# measuring
		for i in range(round(measure_time/dt)):
			state = one_step_integrator(state, ders, dt, inp=p)
			buff[0]=buff[1]; buff[1]=buff[2]; buff[2]=state[0]
			if buff[1]>buff[0] and buff[1]>buff[2]:
				biffd.append(interpolate_max(buff[0], buff[1], buff[2]))
				biffd_args.append(p)
	return biffd_args, biffd

def interpolate_max(v1, v2, v3):
	"""quadratically interpolates the maximum of the three values"""
	a = (v1-2*v2+v3)/2.
	b = (v3-v1)/2.
	if a==0: # avoid division by zero
		return v2 
	return v2-(b**2)/(4*a)

def oscillator_period(ders, initial_condition=None, warmup_time=1000.0, thr=0.0, dt=0.01):
	"""calculates the natural period of the oscillator
	
	:param ders: a list of state variable derivatives
	:param initial_condition: the initial condition (default random)
	:param warmup_time: the time for relaxing to the stable orbit (default 1000)
	:param thr: threshold for determining period (default 0.0)
	:param dt: time step (default 0.01)
	:returns: natural period"""
	# initial conditions
	if initial_condition==None:
		state = [random.gauss(0,0.2) for i in range(len(ders))]
	else:
		state = initial_condition.copy()
	# warmup
	for i in range(round(warmup_time/dt)):
		state = one_step_integrator(state, ders, dt)
	# debuging
	signal = []
	# integration up to x = thr
	xh = state[0]
	while((state[0] > thr and xh < thr) == False):
		xh = state[0]
		state = one_step_integrator(state, ders, dt)
	# Henon trick
	dt_beggining = 1.0/ders[0](state)*(state[0]-thr)
	#print("\tdt_begining = "+str(dt_beggining/dt)+"*dt")
	# spoil condition and go again to x = 0 (still counting time)
	xh = state[0]
	time = 0
	while((state[0] > thr and xh < thr) == False):
		xh = state[0]
		state = one_step_integrator(state, ders, dt)
		time = time + dt
	# another Henon trick
	dt_end = 1.0/ders[0](state)*(state[0]-thr)
	#print("\tdt_end = "+str(dt_end/dt)+"*dt")
	return time + dt_beggining - dt_end


def oscillator_phase(ders, initial_condition, period=None, warmup_time=1000.0, period_counts=5, thr=0.0, dt=0.01):
	"""calculates the asymptotic phase of an initial condition
	
	:param ders: a list of state variable derivatives
	:param initial_condition: the initial condition
	:param period: the period of the oscillator (default None)
	:param warmup_time: the time for relaxing to the stable orbit (default 1000)
	:param period_counts: how many periods to wait for evaluating the asymptotic phase shift (default 5)
	:param thr: threshold for determining period (default 0.0)
	:param dt: time step (default 0.01)
	:returns: asymptotic phase"""
	if period==None:
		period = oscillator_period(ders)
	# initial condition
	state = initial_condition.copy()
	# warmup
	for i in range(round(warmup_time/dt)):
		state = one_step_integrator(state, ders, dt)
	# integration up to x = thr
	time = round(warmup_time/dt)*dt
	xh = state[0]
	while((state[0] > thr and xh < thr) == False):
		xh = state[0]
		state = one_step_integrator(state, ders, dt)
		time += dt
	# Henon trick
	dt_end = 1.0/ders[0](state)*(state[0]-thr)
	time -= dt_end
	# phase calculation
	phase = (time%period)/period
	return phase


def oscillator_PRC(ders, dire=0, initial_condition=None, warmup_time=1000.0, shift=0.01, period_counts=5, dph=0.1, thr=0.0, dt=0.01):
	"""calculates the PRC
	
	:param ders: a list of state variable derivatives
	:param dire: direction of the gradient, x->0,y->1,z->2,... (default 0)
	:param initial_condition: the initial condition (default None)
	:param warmup_time: the time for relaxing to the stable orbit (default 1000)
	:param shift: shift in state space to evaluate the phase gradient (default 0.01)
	:param period_counts: how many periods to wait for evaluating the asymptotic phase shift (default 5)
	:param dph: phase resolution (default 0.1)
	:param thr: threshold for determining period (default 0.0)
	:param dt: time step (default 0.01)
	:returns: phase response curve"""
	period = oscillator_period(ders, thr=thr)
	PRC = [[dph*i for i in range(floor(1.0/dph))],[0 for i in range(floor(1.0/dph))]] # PRC list
	# initial conditions
	if initial_condition==None:
		state = [random.gauss(0,0.2) for i in range(len(ders))]
	else:
		state = initial_condition.copy()
	# warmup
	for i in range(round(warmup_time/dt)):
		state = one_step_integrator(state, ders, dt)
	# stimulating phases
	for ph in [dph*i for i in range(floor(1.0/dph))]:
		# integration up to x = thr
		xh = state[0]
		while((state[0] > thr and xh < thr) == False):
			xh = state[0]
			state = one_step_integrator(state, ders, dt)
		# Henon trick
		dt_beggining = 1.0/ders[0](state)*(state[0]-thr)
		# spoil condition and go to ph (counting time)
		xh = state[0]
		time = dt_beggining
		while(time < ph*period):
			xh = state[0]
			state = one_step_integrator(state, ders, dt)
			time = time + dt
		# shift
		for i in range(len(state)):
			if i==dire:
				state[i] += shift
		#integrate for some periods
		for p in range(period_counts):
			xh = state[0] # spoil
			while((state[0] > thr and xh < thr) == False):
				xh = state[0]
				state = one_step_integrator(state, ders, dt)
				time = time + dt
		# another Henon trick
		dt_end = 1.0/ders[0](state)*(state[0]-thr)
		phase_shift = (time-dt_end - period_counts*period)/period
		PRC[1][round(ph/dph)] = phase_shift/shift
	return PRC
