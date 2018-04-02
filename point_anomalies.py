import pickle
import pandas as pd

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
	f.a = (f.a) | (f.a.notnull() &
			(f[col] < f[col + '_min']) | (f[col] > f[col + '_max']))

a = f[f.a == True][columns]

a.to_csv('point_anomalies.csv', index=False, encoding='utf-8')