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
    # x = np.array(x)

def p(t):
	m,a,b = t
	print("--- Number of states:", m.n_components,"---")
	print("mean:    ", a[0], " vs ", b[0], "\t\tdiff: ", a[0]-b[0])
	print("median:  ", a[1], " vs ", b[1], "\t\tdiff: ", a[1]-b[1])
	print("std:     ", a[2], " vs ", b[2], "\t\tdiff: ", a[2]-b[2])
	print("min:     ", a[3], " vs ", b[3], "\t\tdiff: ", a[3]-b[3])
	print("max:     ", a[4], " vs ", b[4], "\t\tdiff: ", a[4]-b[4])

def p_all():
	for q in x:
		print("\n\n")
		for r in q:
			print("\n\n")
			p(r)

def p_s(n):
	for r in x[n]:
		print("\n\n")
		p(r)

def reformat():
	for q in x:
		arr = np.ar