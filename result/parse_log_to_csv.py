import csv
from glob import glob
import os
import sys

import numpy as np

def run_merge_csvs(result_dir):
	log_filepaths = glob(os.path.join(result_dir,'*.log'))
	nfiles = len(log_filepaths)

	maps = []
	returns = []
	for _ in range(nfiles):
		maps.append(list())
		returns.append(list())

	for idx,log_filepath in enumerate(log_filepaths):
		with open(log_filepath,'r') as f:
			print("Parsing log: {}".format(log_filepath))
			for line in f.readlines():
				if line.startswith('INFO') and "MAP" in line and "Return" in line:
					tokens = line.split()

					MAP = tokens[2]
					Return = tokens[5]

					maps[idx].append(float(MAP))
					returns[idx].append(float(Return))

	header = [ 'epoch' ] + list(range(1,nfiles+1,1)) + ['MAP_average'] + list(range(1,nfiles+1,1)) + ['Return_average']

	epoch_length = len(maps[0])
	# Check Log File Inconsistencies
	for m, r in zip(maps,returns):
		assert len(m) == epoch_length
		assert len(r) == epoch_length


	merged_data = []
	for epoch_idx in range(epoch_length):
		map_data = [ m[epoch_idx] for m in maps ]
		map_avg = np.mean(map_data)

		ret_data = [ r[epoch_idx] for r in returns ]
		ret_avg = np.mean(ret_data)

		row_data = map_data + [ map_avg ] + ret_data + [ ret_avg ]
		merged_data.append( row_data )

	merged_csv = os.path.join(result_dir,'merged.csv')

	with open(merged_csv,'w') as f:
		writer = csv.writer(f,delimiter=',')
		print("Writing merged file to {}".format(merged_csv))
		writer.writerow(header)
		for epoch, row in enumerate(merged_data):
			writer.writerow( [ epoch ] + row )


if __name__ == "__main__":
	result_dir = sys.argv[1]

	run_merge_csvs(result_dir)
