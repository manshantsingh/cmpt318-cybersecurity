import pickle
import pandas as pd
import matplotlib.pyplot as plt

with open('train.pickle', 'rb') as handle:
    train = pickle.load(handle)


with open('test.pickle', 'rb') as handle:
    test = pickle.load(handle)
    test['t'] = test.d.dt.hour * 60 + test.d.dt.minute


g = train.groupby(train.d.dt.hour * 60 + train.d.dt.minute)
x = g.min().join(g.max(), lsuffix='_min', rsuffix='_max')
x['t'] = x.index

f = test.merge(x, on='t')

f['a'] = False

columns = ['Global_active_power', 'Global_reactive_power', 'Voltage',
       		'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
       		'Sub_metering_3']

for col in columns:
	cmin = col + '_min'
	cmax = col + '_max'
	f.a = f.a | (f[col].notnull() & f[cmin].notnull() & f[cmax].notnull() &
			(f[col] < f[cmin]) | (f[col] > f[cmax]))

a = f[f.a == True][columns]

print("percentage: ",100*len(a)/len(test),"%")
print("number of anomalies: ", len(a))


a.to_csv('point_anomalies.csv', index=False, encoding='utf-8')

