import csv
from glob import glob
import os


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
	
	header = [ 'epoch' ] + list(range(1,11,1)) + list(range(1,11,1))

	merged_data = []
	for d in all_data:
		MAPs = d[0::2]
		Rets = d[1::2]
		merged_data.append( MAPs + Rets )

	merged_csv = result_dir + '_merged.csv'
	with open(merged_csv,'w') as f:
		writer = csv.writer(f,delimiter=',')
		print("Writing merged file to {}".format(merged_csv))
		writer.writerow(header)
		for epoch, row in enumerate(merged_data):
			writer.writerow( [ epoch ] + row )	


if __name__ == "__main__":
	result_dir = './result/wig/onebest_CMVN'
	
	run_merge_csvs(result_dir)
