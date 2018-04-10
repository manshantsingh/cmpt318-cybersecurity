import pickle
from sys import argv
import numpy as np
import pandas as pd
import sys

import warnings
warnings.filterwarnings(action='ignore')
# warnings.filterwarnings("ignore",category=DeprecationWarning)


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


columns_to_use = ['Voltage']

def get_pickle_object(fileName):
	with open(fileName, 'rb') as handle:
	    return pickle.load(handle)

def generate_in_format(fileName):
	x = get_pickle_object(fileName)
	# another.append(x)
	arr = []
	for a in x:
		arr.append((a[columns_to_use].values, a.d.iloc[0]))
	return arr
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
	test_file_name = 'test-'+window+'.pickle'
	model_file_name = 'models/model-'+window+'.pickle'
	test = generate_in_format(test_file_name)
	model = get_pickle_object(model_file_name)

	x = score(model[0], test).set_index('date')

	l = x.logprob
	mean = l.mean()
	std = l.std()
	std_diff = np.abs(l - mean)/std

	x['cutoff by min_in_train'] = l < model[1]
	x['anomaly 68% confidence'] = std_diff > 1
	x['anomaly 95% confidence'] = std_diff > 2
	x['anomaly 99.7% confidence'] = std_diff > 3

	print("\n\n\nwindows:", window, "("+str(l.count())+")")
	print(x.drop(columns=['logprob']).sum())

	x.to_csv('csv/anomalies-'+window+'.csv', index=False, encoding='utf-8')
