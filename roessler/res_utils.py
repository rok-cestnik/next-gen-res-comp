from reservoir import *
from utils import interpolate_max

def res_one_step(seq, uH, w):
	r = reservoir(seq, uH)
	new_u = sum(r[j]*w[j] for j in range(len(w)))
	new_uH = uH[new_u.shape[1]:]+[float(new_u[0,n]) for n in range(new_u.shape[1])]
	return new_uH

def res_integrate_signal(seq, w, uH, steps):
	signal = [[uH[-w.shape[1]+i] for i in range(w.shape[1])]]
	for i in range(steps):
		if(i%50 == 0):
			print("\t", i, "/", steps)
		uH = res_one_step(seq, uH, w)
		signal.append([uH[-w.shape[1]+i] for i in range(w.shape[1])])
	return signal

def res_bifurcation_diagram(seq, w, uH, comp=0, p_range=[0.5,0.9], p_res=0.01, warmup_steps=500, small_warmup=100, measure_steps=300):
	biffd_args = []
	biffd = []
	index = -w.shape[1]+comp
	# warmup
	buff = [0,0,0] # 3 valued signal buffer
	for i in range(warmup_steps):
		uH = res_one_step(seq, uH, w)
		uH[-1] = p_range[0]
		buff[0]=buff[1]; buff[1]=buff[2]; buff[2]=uH[index]
	# bifurcation diagram
	p_vals = [p_range[0]+p_res*i for i in range(round((p_range[1]-p_range[0])/p_res))]
	for p in p_vals:
		print("\tp = ", p, "/", p_range[1])
		# small warmup
		for i in range(small_warmup):
			uH = res_one_step(seq, uH, w)
			uH[-1] = p
			buff[0]=buff[1]; buff[1]=buff[2]; buff[2]=uH[index]
		# measuring
		for i in range(measure_steps):
			uH = res_one_step(seq, uH, w)
			uH[-1] = p
			buff[0]=buff[1]; buff[1]=buff[2]; buff[2]=uH[index]
			if buff[1]>buff[0] and buff[1]>buff[2]:
				biffd.append(interpolate_max(buff[0], buff[1], buff[2]))
				biffd_args.append(p)
	return biffd_args, biffd

def res_estimate_period(seq, w, uH, comp=0, warmup_steps=1000, thr=0.5, periods=5):
	# warmup
	for i in range(warmup_steps):
		uH = res_one_step(seq, uH, w)
	# integration up to x = thr
	index = -w.shape[1]+comp
	xp = uH[index]
	while((uH[index] > thr and xp < thr) == False):
		xp = uH[index]
		uH = res_one_step(seq, uH, w)
	# estimate how much over threshold it went
	time = (uH[index]-thr)/(uH[index]-xp)
	for p in range(periods):
		# spoil condition and go again to x = thr (still counting time)
		xp = uH[index]
		while((uH[index] > thr and xp < thr) == False):
			xp = uH[index]
			uH = uH = res_one_step(seq, uH, w)
			time += 1
	# estimate how much over threshold it went again
	time -= (uH[index]-thr)/(uH[index]-xp)
	return time/periods

def res_phase(seq, w, uH, period=False, comp=0, warmup_steps=1000, thr=0.5):
	if period==False:
		period = res_estimate_period(seq, w, uH.copy(), thr=thr)
	# warmup (start counting time)
	time = 0
	for i in range(warmup_steps):
		uH = res_one_step(seq, uH, w)
		time += 1
	# integration up to x = thr
	index = -w.shape[1]+comp
	xp = uH[index]
	while((uH[index] > thr and xp < thr) == False):
		xp = uH[index]
		uH = uH = res_one_step(seq, uH, w)
		time += 1
	# estimate how much over threshold it went
	time -= (uH[index]-thr)/(uH[index]-xp)
	return (time%period)/period

def res_PRC(seq, w, uH, period=False, comp=0, warmup_steps=1000, shift=0.1, thr=0.5):
	print("computing the phase response curve:")
	if period==False:
		period = res_estimate_period(seq, w, uH.copy(), thr=thr)
	# warmup to limit cycle
	for i in range(warmup_steps):
		uH = res_one_step(seq, uH, w)
	# integration up to x = thr
	index = -w.shape[1]+comp
	xp = uH[index]
	while((uH[index] > thr and xp < thr) == False):
		xp = uH[index]
		uH = uH = res_one_step(seq, uH, w)
	# measure phase responses
	PRC = [[],[]]
	for r in range(int(period)):
		print("\tr = "+str(r)+"/"+str(int(period)))
		uHs = uH.copy()
		uHs[index] += shift
		phase_LC = res_phase(seq, w, uH.copy(), period=period, thr=thr)
		phase_shifted = res_phase(seq, w, uHs, period=period, thr=thr)
		PRC[0].append(phase_LC)
		PRC[1].append((phase_shifted-phase_LC)/shift)
		uH = res_one_step(seq, uH, w)
	return PRC
	
	
