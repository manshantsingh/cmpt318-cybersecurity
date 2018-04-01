import pickle
import sys
import numpy as np

if len(sys.argv) > 1:
	current_file_in_use = sys.argv[1]
else:
	current_file_in_use = 'train.pickle'
print("using pickle file:", current_file_in_use)


with open(current_file_in_use, 'rb') as handle:
    x = pickle.load(handle)

if len(sys.argv) > 2 and sys.argv[2]=='m':
	x = np.array(x)

def p(t):
	m,a,b = t
	print("model:", m)
	print("mean:    ", a[0], " vs ", b[0])
	print("median:  ", a[1], " vs ", b[1])
	print("std:     ", a[2], " vs ", b[2])
	print("min:     ", a[3], " vs ", b[3])
	print("max:     ", a[4], " vs ", b[4])

def p_all():
	for q in x:
		for r in q:
			print("\n\n")
			p(r)

def p_s(n):
	for r in x[n]:
		print("\n\n")
		p(r)