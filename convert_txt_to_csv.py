__author__ = 'mkv-aql'
import pandas as pd

read_file = pd.read_csv (r'C:/Users/Makav/Desktop/ISM_2022w/train.txt')
read_file.to_csv (r'C:/Users/Makav/Desktop/ISM_2022w/train.csv', index=None)

