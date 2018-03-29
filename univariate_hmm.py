import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


with open('../train-friday_night.pickle', 'rb') as handle:
    x = pickle.load(handle)

arr = []
for a in x:
	v = a.set_index(['d']).Global_active_power.values
	if len(v) == 1440:
		arr.append(v[1080:])

arr = np.array(arr, dtype=np.float64)

def score(m, a):
	val = []
	for v in a:
		val.append(m.score([v]))
	val = np.array(val)
	return [val.mean(), val.std(), val.min(), val.max()]


def wow(n,f=5,nn=50):
	k = KFold(f)
	for a_ind, b_ind in k.split(arr):
		a = arr[a_ind]
		b = arr[b_ind]
		m = hmm.GaussianHMM(n_components=n, n_iter=nn)
		m.fit(a)
		a = score(m,a)
		b = score(m,b)
		print("\n\n train vs test")
		print("mean:    ", a[0], " vs ", b[0])
		print("std:     ", a[1], " vs ", b[1])
		print("min:     ", a[2], " vs ", b[2])
		print("max:     ", a[3], " vs ", b[3])