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
	print("mean:    ", a[0], " vs ", b[0], "\t\tdiff: ", a[0]-b[0], "\t\tvalidate: ", c[0], "\t\tdiff: ", a[0]-c[0])
	print("median:  ", a[1], " vs ", b[1], "\t\tdiff: ", a[1]-b[1], "\t\tvalidate: ", c[1], "\t\tdiff: ", a[1]-c[1])
	print("std:     ", a[2], " vs ", b[2], "\t\tdiff: ", a[2]-b[2], "\t\tvalidate: ", c[2], "\t\tdiff: ", a[2]-c[2])
	print("min:     ", a[3], " vs ", b[3], "\t\tdiff: ", a[3]-b[3], "\t\tvalidate: ", c[3], "\t\tdiff: ", a[3]-c[3])
	print("max:     ", a[4], " vs ", b[4], "\t\tdiff: ", a[4]-b[4], "\t\tvalidate: ", c[4], "\t\tdiff: ", a[4]-c[4])

	total = 0
	limit = a[3]
	for a in c[5]:
		if a < limit:
			total+=1

	print("anomalies: ", total, "\t\tnumber of points: ", len(c[5]), "\t\tpercentage of anomalies: ", 100*total/len(c[5]), "%")


def p_all(arr=None):
	for r in x:
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


def dump(fileName, variable):
	with open(fileName+".pickle", 'wb') as handle:
		pickle.dump(variable, handle, protocol=pickle.HIGHEST_PROTOCOL)

if len(sys.argv) > 2 and sys.argv[2]=='all':
	x = np.array(x)
	p_all()

def model(a, fileName):
	dump("models/"+fileName, (a[0], a[1][3]))