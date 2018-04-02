import pickle
import sys
import numpy as np

import warnings
# warnings.filterwarnings(action='once')
warnings.filterwarnings(action='ignore')

if len(sys.argv) > 1:
	current_file_in_use = sys.argv[1]
else:
	current_file_in_use = 'train.pickle'
print("using pickle file:", current_file_in_use)


with open(current_file_in_use, 'rb') as handle:
    x = pickle.load(handle)
    # x = np.array(x)

def p(t, arr=None):
	m,a,b,c = t
	print("--- Number of states:", m.n_components,"---")
	# c = score(m, arr)
	print("mean:    ", a[0], " vs ", b[0], "\t\tdiff: ", a[0]-b[0], "\t\t: ", c[0])
	print("median:  ", a[1], " vs ", b[1], "\t\tdiff: ", a[1]-b[1], "\t\t: ", c[1])
	print("std:     ", a[2], " vs ", b[2], "\t\tdiff: ", a[2]-b[2], "\t\t: ", c[2])
	print("min:     ", a[3], " vs ", b[3], "\t\tdiff: ", a[3]-b[3], "\t\t: ", c[3])
	print("max:     ", a[4], " vs ", b[4], "\t\tdiff: ", a[4]-b[4], "\t\t: ", c[4])


def p_all(arr=None):
	for q in x:
		print("\n\n")
		for r in q:
			print("\n\n")
			p(r, arr)

def p_s(n, arr=None):
	for r in x[n]:
		print("\n\n")
		p(r,arr)


def reform():
	arr = []
	for a in x:
		arr.append(a.set_index(['d'])[['Global_active_power']].values)

	return np.array(arr, dtype=np.float64)

def score(m, a):
	val = []
	for v in a:
		val.append(m.score(v))
	val = np.array(val)
	return [val.mean(), np.median(val), val.std(), val.min(), val.max()]