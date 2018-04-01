import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
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

# change me
columns_to_use = ['Global_active_power', 'Global_reactive_power', 'Voltage',
			       'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
			       'Sub_metering_3']

with open(current_file_in_use, 'rb') as handle:
    x = pickle.load(handle)

arr = []
for a in x:
	arr.append(a.set_index(['d'])[columns_to_use].values)

arr = np.array(arr, dtype=np.float64)

def score(m, a):
	val = []
	for v in a:
		val.append(m.score(v))
	val = np.array(val)
	return [val.mean(), np.median(val), val.std(), val.min(), val.max()]

all_models = []

def wow(n_components, n_folds=5, n_iter=50):
	k = KFold(n_splits=n_folds, shuffle=True)

	all_models.append([])

	for a_ind, b_ind in k.split(arr):

		# train
		a_orig = arr[a_ind]
		a_len = np.full(len(a_orig), len(a_orig[0]))
		a = a_orig.reshape(-1, len(columns_to_use))

		# test
		b_orig = arr[b_ind]
		b_len = np.full(len(b_orig), len(b_orig[0]))
		b = b_orig.reshape(-1, len(columns_to_use))

		# fitting the model
		m = hmm.GaussianHMM(n_components=n_components, n_iter=n_iter)
		m.fit(a, a_len)

		a = score(m,a_orig)
		b = score(m,b_orig)
		print("train(",len(a_orig),") vs test(",len(b_orig),")")
		print("mean:    ", a[0], " vs ", b[0])
		print("median:  ", a[1], " vs ", b[1])
		print("std:     ", a[2], " vs ", b[2])
		print("min:     ", a[3], " vs ", b[3])
		print("max:     ", a[4], " vs ", b[4])
		all_models[-1].append((m,a,b))

def dump(fileName, variable=np.array(all_models)):
	with open(fileName+"-"+current_file_in_use+".pickle", 'wb') as handle:
		pickle.dump(variable, handle, protocol=pickle.HIGHEST_PROTOCOL)

