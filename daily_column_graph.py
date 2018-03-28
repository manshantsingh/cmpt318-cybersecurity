import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.model_selection import train_test_split

combined = 'c' in sys.argv
smoothed = 's' in sys.argv
do_deviation = 'std' in sys.argv

with open('train.pickle', 'rb') as handle:
    df = pickle.load(handle)

if smoothed:
	df = df.set_index('d').rolling(7).mean().dropna(0).reset_index()
	# df = df.set_index('d').rolling('1h', min_periods=60).mean().dropna(0).reset_index()

def func(x, d, legend, col, ax):
	x = x[x.d.dt.weekday == d]
	x['t'] = x.d.dt.hour * 60 + x.d.dt.minute
	g = x.groupby(x.t)
	if do_deviation:
		m = g.std().reset_index()
	else:
		m = g.mean().reset_index()
	ax.plot(m.t/60, m[col])
	legend.append(x.iloc[0].d.weekday_name)


columns = ['Global_active_power', 'Global_reactive_power', 'Voltage',
	       'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
	       'Sub_metering_3']
columns = ['Global_active_power']

for col in columns:
	title = 'daily '+col
	if smoothed:
		title += ' (smoothed)'
	if do_deviation:
		title += ' standard deviation'
	plt.clf()
	plt.cla()
	plt.close()
	if combined:
		legend = []
		for i in range(7):
			func(df, i, legend, col, plt)
		plt.legend(legend)
		plt.title(title)
		plt.savefig('temp/'+title+'.png')
	else:
		f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(30,15))
		# weekdays
		legend = []
		for i in range(5):
			func(df, i, legend, col, ax1)
		ax1.set_title(title+' (weekdays)', fontsize=20)
		ax1.legend(legend)
		# weekends
		legend = []
		for i in range(4,7):
			func(df, i, legend, col, ax2)
		ax2.legend(legend)
		ax2.set_title(title+' (weekend)', fontsize=20)
		plt.savefig('temp/'+title+'2.png')
	print("done: "+title)