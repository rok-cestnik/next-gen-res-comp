from math import sqrt, tanh, cosh, floor
import numpy as np
import random
import copy
from matplotlib import pyplot


# parameters
M = 6000

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
