from sys import argv
import pickle
import pandas as pd
import matplotlib.pyplot as plt


if len(argv)>1 and argv[1].endswith('.pickle'):
	pickleFileName = argv[1]
else:
	pickleFileName = 'train.pickle'

combined = 'c' in argv
smoothed = 's' in argv
do_deviation = 'std' in argv
do_median  = 'median' in argv

titlePrefix = pickleFileName[:-len('.pickle')]

with open(pickleFileName, 'rb') as handle:
    df = pickle.load(handle)

if smoothed:
	df = df[(df.d.dt.month > 3) & (df.d.dt.month < 10)]
	df = df.set_index('d').rolling(15).mean().dropna(0).reset_index()
	# df = df.set_index('d').rolling('1h', min_periods=60).mean().dropna(0).reset_index()

def func(x, d, legend, col, ax):
	x = x[x.d.dt.weekday == d]
	x['t'] = x.d.dt.hour * 60 + x.d.dt.minute
	g = x.groupby(x.t)
	if do_median:
		m = g.median().reset_index()
	elif do_deviation:
		m = g.std().reset_index()
	else:
		m = g.mean().reset_index()

	m['o'] = m.t//30
	m=m.groupby(m.o).mean()

	ax.plot(m.t/60, m[col])
	legend.append(x.iloc[0].d.weekday_name)


# columns = ['Global_active_power', 'Global_reactive_power', 'Voltage',
# 	       'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
# 	       'Sub_metering_3']
columns = ['Voltage']

for col in columns:
	title = titlePrefix + ' '+col
	if smoothed:
		title += ' (smoothed)'
	if do_median:
		title += ' median'
	elif do_deviation:
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