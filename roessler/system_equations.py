import numpy as np
from math import tanh, cosh

def get_ders(system):
	if(system == 'roessler'):
		a = 0.2
		b = 0.4
		c = 5.7
		def dx(state, inp=0):
			return -state[1] - state[2]
		def dy(state, inp=0):
			return state[0] + a*state[1]
		def dz(state, inp=0):
			return b + state[2]*(state[0]-c)
		ders = [dx,dy,dz]
	if(system == 'roessler_input'):
		a = 0.2
		b = 0.4#1.6
		c = 5.7
		def dx(state, inp=0):
			return -state[1] - state[2]
		def dy(state, inp=0):
			return state[0] + a*state[1] + inp
		def dz(state, inp=0):
			return b + state[2]*(state[0]-c)
		ders = [dx,dy,dz]
	elif(system == 'lorenz'):
		sigma = 10
		rho = 28
		beta = 8/3.0
		def dx(state, inp=0):
			return sigma*(state[1] - state[0])
		def dy(state, inp=0):
			return state[0]*(rho - state[2]) - state[1]
		def dz(state, inp=0):
			return state[0]*state[1] - beta*state[2]
		ders = [dx,dy,dz]
	elif(system == 'morris-lecar'):
		I = 0.07
		gL = 0.5
		gK = 2
		gCa = 1.33
		V1 = -0.01
		V2 = 0.15
		V3 = 0.1
		V4 = 0.145
		VL = -0.5
		VK = -0.7
		VCa = 1
		def mi(V):
			return (1+tanh((V-V1)/V2))/2
		def wi(V):
			return (1+tanh((V-V3)/V4))/2
		def lam(V):
			return cosh((V-V3)/(2*V4))/3
		def dV(state, inp=0):
			return I + state[2] - gL*(state[0]-VL) - gK*state[1]*(state[0]-VK) - gCa*mi(state[0])*(state[0]-VCa)
		def dw(state, inp=0):
			return lam(state[0])*(wi(state[0])-state[1])
		ders = [dV,dw]
	return ders
