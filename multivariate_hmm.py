import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
from math import log
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import warnings
# warnings.filterwarnings(action='once')
warnings.filterwarnings(action='ignore')

if len(sys.argv) > 1:
	current_file_in_use = sys.argv[1]
else:
	current_file_in_use = 'train-friday_night.pickle'
print("using pickle file:", current_file_in_use)

if len(sys.argv) > 2:
	current_test_file_in_use = sys.argv[2]
	print("using test pickle file:", current_test_file_in_use)

# change me
# columns_to_use = ['Global_active_power', 'Global_reactive_power', 'Voltage',
# 			       'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
# 			       'Sub_metering_3']

columns_to_use = ['Voltage']
another=[]
def generate_in_format(fileName):
	with open(fileName, 'rb') as handle:
	    x = pickle.load(handle)
	another.append(x)
	arr = []
	for a in x:
		arr.append(a[columns_to_use].values)

	# # for i in range(len(arr)-1,-1,-1):
	# for i in range(len(arr)):
	# 	a = arr[i]
	# 	plt.clf()
	# 	plt.cla()    
	# 	plt.close()
	# 	plt.plot(np.arange(len(a)), a)
	# 	# plt.title(str(a.d.dt.date.values[0]) + "  " + str(a.d.dt.weekday_name.values[0]))
	# 	plt.show()
	return np.array(arr, dtype=np.float64)

arr = generate_in_format(current_file_in_use)
if current_test_file_in_use != None:
	test_arr = generate_in_format(current_test_file_in_use)

def score(m, a):
	val = []
	for v in a:
		val.append(m.score(v))
	val = np.array(val)
	return [val.mean(), np.median(val), val.std(), val.min(), val.max(), val]

all_models = []

# def wow(n_components, n_folds=5, n_iter=50):
# 	k = KFold(n_splits=n_folds, shuffle=False)

# 	all_models.append([])

# 	for a_ind, b_ind in k.split(arr):

# 		# train
# 		a_orig = arr[a_ind]
# 		a_len = np.full(len(a_orig), len(a_orig[0]))
# 		a = a_orig.reshape(-1, len(columns_to_use))

# 		# test
# 		b_orig = arr[b_ind]
# 		b_len = np.full(len(b_orig), len(b_orig[0]))
# 		b = b_orig.reshape(-1, len(columns_to_use))

# 		# fitting the model
# 		m = hmm.GaussianHMM(n_components=n_components, n_iter=n_iter)
# 		m.fit(a, a_len)

# 		a = score(m,a_orig)
# 		b = score(m,b_orig)
# 		if current_test_file_in_use != None:
# 			c = score(m, test_arr)
# 		else:
# 			c = ['-' for i in range(5)]
# 		print("train(",len(a_orig),") vs validate(",len(b_orig),")\t\tand test_arr(",len(test_arr),")")
# 		print("mean:    ", a[0], " vs ", b[0], "\t\tdiff: ", a[0]-b[0], "\t\t: ", c[0])
# 		print("median:  ", a[1], " vs ", b[1], "\t\tdiff: ", a[1]-b[1], "\t\t: ", c[1])
# 		print("std:     ", a[2], " vs ", b[2], "\t\tdiff: ", a[2]-b[2], "\t\t: ", c[2])
# 		print("min:     ", a[3], " vs ", b[3], "\t\tdiff: ", a[3]-b[3], "\t\t: ", c[3])
# 		print("max:     ", a[4], " vs ", b[4], "\t\tdiff: ", a[4]-b[4], "\t\t: ", c[4])
# 		print("BIC value="+str(-2*a[0] + len(columns_to_use)*log(n_components)))
# 		all_models[-1].append((m,a,b,c))

def hehe(n_components, n_iter=1000):
	a_orig, b_orig = train_test_split(arr, shuffle=False)
	a_len = np.full(len(a_orig), len(a_orig[0]))
	a = a_orig.reshape(-1, len(columns_to_use))

	# test
	# b_orig = arr[b_ind]
	b_len = np.full(len(b_orig), len(b_orig[0]))
	b = b_orig.reshape(-1, len(columns_to_use))

	# fitting the model
	m = hmm.GaussianHMM(n_components=n_components, n_iter=n_iter, tol=1, verbose=False)
	m.fit(a, a_len)

	a = score(m,a_orig)
	b = score(m,b_orig)
	if current_test_file_in_use != None:
		c = score(m, test_arr)
	else:
		c = ['-' for i in range(5)]
	print("train(",len(a_orig),") vs validate(",len(b_orig),")\t\tand test_arr(",len(test_arr),")")
	print("mean:    ", a[0], " vs ", b[0], "\t\tdiff: ", a[0]-b[0], "\t\tvalidate: ", c[0], "\t\tdiff: ", a[0]-c[0])
	print("median:  ", a[1], " vs ", b[1], "\t\tdiff: ", a[1]-b[1], "\t\tvalidate: ", c[1], "\t\tdiff: ", a[1]-c[1])
	print("std:     ", a[2], " vs ", b[2], "\t\tdiff: ", a[2]-b[2], "\t\tvalidate: ", c[2], "\t\tdiff: ", a[2]-c[2])
	print("min:     ", a[3], " vs ", b[3], "\t\tdiff: ", a[3]-b[3], "\t\tvalidate: ", c[3], "\t\tdiff: ", a[3]-c[3])
	print("max:     ", a[4], " vs ", b[4], "\t\tdiff: ", a[4]-b[4], "\t\tvalidate: ", c[4], "\t\tdiff: ", a[4]-c[4])
	print("BIC value="+str(-2*a[0] + len(columns_to_use)*log(len(a_orig))))
	
	title = current_file_in_use[len('train-') : -len('.pickle')] + "with " + str(n_components) + " hidden states"

	plt.clf()
	plt.cla()    
	plt.close()
	f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(8,5))
	ax1.boxplot(a[5])
	ax2.boxplot(b[5])
	ax3.boxplot(c[5])
	ax1.set_ylabel('Log Likelihood')
	ax1.set_xlabel('train (fit) dataset')
	ax2.set_xlabel('train (validate) dataset')
	ax3.set_xlabel('test dataset')
	
	plt.title(title,x=-2/3.0)
	# plt.tight_layout()
	# plt.show()
	plt.savefig('temp/'+title+'.png')

	all_models.append((m,a,b,c))

def dump(fileName, variable):
	with open(fileName+"-"+current_file_in_use+".pickle", 'wb') as handle:
		pickle.dump(variable, handle, protocol=pickle.HIGHEST_PROTOCOL)

# def dump(fileName):
# 	dump(fileName, np.array(all_models))


def loop(start=3, end_limit=15, store=False):
	for i in range(start,end_limit):
		print("\n\n--- starting n_components="+str(i)+" ---")
		hehe(i)
		if store:
			dump("alone-"+str(i), all_models[-1])
