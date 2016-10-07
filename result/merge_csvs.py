import csv
from glob import glob
import os

import numpy as np

def run_merge_csvs(result_dir):
	all_data = []
	for fold in range(1,11,1):
		csv_filepath = result_dir + "_fold{}".format(fold) + '.csv'
		print(csv_filepath)
		with open(csv_filepath,'r') as f:
			reader = csv.reader(f)
			data = list(reader)
			if fold == 1:
				all_data = data
			else:
				for idx, d in enumerate(data):
					all_data[idx] += d
	
	print("len(all_data[0]) {}".format(len(all_data[0])))
	
	header = [ 'epoch' ] + list(range(1,11,1)) + ['MAP_average'] + list(range(1,11,1)) + ['Return_average']

	merged_data = []
	for d in all_data:
		MAPs = list(map(float,d[0::2]))

		MAP_avg = np.mean(MAPs)
	
		MAPs += [None] * ( 10 - len(MAPs) )  

		Rets = list(map(float,d[1::2]))
		
		Ret_avg = np.mean(Rets)
		
		Rets += [None] * ( 10 - len(Rets) )  
		
		row_data = MAPs + [ MAP_avg ] + Rets + [ Ret_avg ] 

		merged_data.append( row_data )

	merged_csv = result_dir + '_merged.csv'
	with open(merged_csv,'w') as f:
		writer = csv.writer(f,delimiter=',')
		print("Writing merged file to {}".format(merged_csv))
		writer.writerow(header)
		for epoch, row in enumerate(merged_data):
			writer.writerow( [ epoch ] + row )	


if __name__ == "__main__":
	result_dir = './result/nqc/onebest_CMVN'
	
	run_merge_csvs(result_dir)
