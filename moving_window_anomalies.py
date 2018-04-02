import pickle
import pandas as pd
from sys import argv

if len(argv) > 1:
	windowSize = int(argv[1])
else:
	windowSize = 13

if len(argv) > 2:
	standardDeviationCuttoff = float(argv[2])
else:
	standardDeviationCuttoff = 3.0

with open('test.pickle', 'rb') as handle:
    x = pickle.load(handle).set_index('d')

r = x.rolling(windowSize, center=True)
m = r.mean()
s = r.std()

w = m.join(s, lsuffix='_mean', rsuffix='_std', how='inner')

f = x.join(w, how='left')

f['a'] = False

columns = ['Global_active_power', 'Global_reactive_power', 'Voltage',
       		'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
       		'Sub_metering_3']

for col in columns:
	mean = col + '_mean'
	std = col + '_std'
	f.a = f.a | (f[col].notnull() & f[mean].notnull() & f[std].notnull() &
			(abs(f[col]-f[mean]) > standardDeviationCuttoff * f[std] ))

a = f[f.a == True][columns]

print("percentage: ",100*len(a)/len(x),"%")
print("number of anomalies: ", len(a))

a.to_csv('moving_window_anomalies.csv', index=False, encoding='utf-8')