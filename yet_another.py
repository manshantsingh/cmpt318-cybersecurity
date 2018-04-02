from sys import argv
import pickle
import pandas as pd
import matplotlib.pyplot as plt

if len(argv)>1 and argv[1].endswith('.pickle'):
	pickleFileName = argv[1]
else:
	pickleFileName = 'test.pickle'

arr = []
for a in g:
	arr.append(a[1])

for i in range(len(arr)-1,-1,-1):
	a=arr[i]
	plt.clf()
	plt.cla()    
	plt.close()
	plt.plot(a.d.dt.hour + a.d.dt.minute/60, a.Global_active_power)
	plt.title(str(a.d.dt.date.values[0]) + "  " + str(a.d.dt.weekday_name.values[0]))
	plt.show()