import pickle
import pandas as pd

def generate_pickle(func, name, file):
	with open(file+'.pickle', 'rb') as handle:
    	x = pickle.load(handle)

    x = func(x)
    with open(file+'-'+name+'.pickle', 'wb') as handle:
		pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

def generate_both_pickle(func, name):
	generate_pickle(func, name, 'train')
	generate_pickle(func, name, 'test')

def friday_night(df):
	