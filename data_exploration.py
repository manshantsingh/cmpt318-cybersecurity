import pandas as pd
import matplotlib.pyplot as plt

# call this function with 'input to group by' and 'output to present'
def generate_graph(df, x, y):
	a=df.groupby(x).mean().reset_index()
	plt.plot(a[x], a[y], 'b.')
	plt.show()

x = pd.read_csv('test.txt')
t = pd.to_datetime(x.Time, format='%H:%M:%S').dt

x['t'] = t.hour * 60 + t.minute
x['d'] = pd.to_datetime(x.Date, format='%d/%m/%Y')

# g = x.groupby(x.t)

# m = g.mean().reset_index()
# plt.plot(m.t/60, m.Voltage)
# plt.show()

# s = g.std().reset_index()
# plt.plot(s.t/60, s.Voltage)
# plt.show()
