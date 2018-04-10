import sys
import pickle
import pandas as pd

def get_pickle_ready(filename):
	x = pd.read_csv(filename+'.txt')
	x['d'] = pd.to_datetime(x.Date + ' ' + x.Time, format='%d/%m/%Y %H:%M:%S')
	x.drop(['Date', 'Time'], axis=1, inplace=True)
	x.dropna(0)
	# x['t'] = x.d.dt.hour * 60 + x.d.dt.minute
	with open(filename+'.pickle', 'wb') as handle:
		pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

# get_pickle_ready('train')
# get_pickle_ready('test')
get_pickle_ready('test2')