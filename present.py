import pickle
from sys import argv
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='ignore')
# warnings.filterwarnings("ignore",category=DeprecationWarning)

if 'old' in sys.argv:
	test_prefex = 'test-'
	csv_folder = 'csv/'
else:
	test_prefex = 'test2-'
	csv_folder = 'csv2/'



all_of_them = [
				'summer-day-weekday',
				'summer-night-weekday',
				'summer-day-weekend',
				'summer-night-weekend',
				'winter-day-weekday',
				'winter-night-weekday',
				'winter-day-weekend',
				'winter-night-weekend'
				]

all_of_them = ['winter-night-weekend']


columns_to_use = ['Voltage']
map_of_all = []

def get_pickle_object(fileName):
	with open(fileName, 'rb') as handle:
	    return pickle.load(handle)

def generate_in_format(fileName):
	x = get_pickle_object(fileName)
	# another.append(x)
	arr = []
	m = {}
	for a in x:
		arr.append((a[columns_to_use].values, a.iloc[0].d))
		m[arr[-1][1]] = arr[-1][0]


	# for i in range(len(arr)):
	# 	a = arr[i][0]
	# 	plt.clf()
	# 	plt.cla()    
	# 	plt.close()
	# 	plt.plot(np.arange(len(a)), a)
	# 	plt.title(str(arr[i][1]))
	# 	plt.show()
	return arr, m
	# return np.array(arr, dtype=np.float64)

def score(m, a):
	val = []
	dates = []
	for v in a:
		val.append(m.score(v[0]))
		dates.append(v[1])
	val = np.array(val)
	return pd.DataFrame({'date':dates, 'logprob':val})

for window in all_of_them:
	test_file_name = test_prefex+window+'.pickle'
	model_file_name = 'models/model-'+window+'.pickle'

	test, m = generate_in_format(test_file_name)
	model = get_pickle_object(model_file_name)

	x = score(model[0], test)
	# .set_index('date')

	l = x.logprob
	mean = l.mean()
	std = l.std()
	std_diff = np.abs(l - mean)/std

	x['cutoff by min_in_train'] = l < model[1]
	x['anomaly 68% confidence'] = std_diff > 1
	x['anomaly 95% confidence'] = std_diff > 2
	x['anomaly 99.7% confidence'] = std_diff > 3

	print("\n\n\nwindows:", window, "("+str(l.count())+")")
	print(x.drop(columns=['logprob', 'date']).sum())

	x.to_csv(csv_folder+'anomalies-'+window+'.csv', index=False, encoding='utf-8')

	x = x.sort_values(by=['logprob'])

	map_of_all.append((x,m))

def plot(df, i, dictionary):
	d = df.iloc[i].date
	a = dictionary[d]
	plt.clf()
	plt.cla()    
	plt.close()
	plt.plot(np.arange(len(a)), a)
	plt.title(str(d))
	plt.show()
