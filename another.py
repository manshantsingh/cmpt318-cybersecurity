import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.model_selection import train_test_split


with open('train-day_1.pickle', 'rb') as handle:
    x = pickle.load(handle)


for i in range(0,len(x), 5):
	plt.clf()
	plt.cla()
	plt.close()
	legend = []
	for j in range(i, i+5, 1):
		plt.plot(x[j].d.dt.hour * 60 + x[j].d.dt.minute, x[j].Global_active_power)
		legend.append(str(x[j].d.values[0]))
	plt.legend(legend)
	plt.show()














# arr = []
# # lengths = []
# for a in x:
# 	v = a.set_index(['d']).Global_active_power.values
# 	if len(v) == 1440:
# 		arr.append(v[700:750])
# 		# lengths.append(1440)
# arr = np.array(arr, dtype=np.float64)
# arr = np.array([a.set_index(['d']).Global_active_power.values for a in x],dtype=np.float64)

# def tt(n, iters=100):
# 	# m = hmm.GaussianHMM(n_components=n, covariance_type="full", n_iter=iters)
# 	# global arr
# 	m = hmm.GaussianHMM(n_components=n, n_iter=iters)
# 	m.fit(arr)
# 	val = np.array([m.score([v]) for v in arr])
# 	print("\n\nn =", n, file=open("output.txt", "a"))
# 	print("val: mean =",val.mean(),"std =", val.std(), file=open("output.txt", "a"))
# 	print("overal score =", m.score(arr), file=open("output.txt", "a"))
# 	# return m

# for i in range(3, 20):
# 	# print("type of i:", type(i))
# 	tt(i)
# # tt(1)

# def wow(n,nn=50, times=10):
# 	val = []
# 	for i in range(times):
# 		a,b = train_test_split(arr)
# 		m = hmm.GaussianHMM(n_components=n, n_iter=nn)
# 		m.fit(a)
# 		val.append(m.score(b))
# 	val=np.array(val)
# 	print("score:",val.mean())