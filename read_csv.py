import pandas as pd
from sys import argv

x = pd.read_csv(argv[1])
x = x.sort_values(by=['logprob'])