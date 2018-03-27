import pickle
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.model_selection import train_test_split


with open('train.pickle', 'rb') as handle:
    df = pickle.load(handle)

def func(x, d, legend, col):
	x = x[x.d.dt.weekday == d]
	x['t'] = x.d.dt.hour * 60 + x.d.dt.minute
	g = x.groupby(x.t)

	m = g.mean().reset_index()
	plt.plot(m.t/60, m[col])
	legend.append(x.iloc[0].d.weekday_name)


columns = ['Global_active_power', 'Global_reactive_power', 'Voltage',
	       'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
	       'Sub_metering_3']

for col in columns:
	plt.clf()
	plt.cla()
	plt.close()

	legend = []
	for i in range(7):
		func(df, i, legend, col)
	title = 'daily '+col
	plt.legend(legend)
	plt.title(title)
	plt.savefig('temp/'+title+'.png')