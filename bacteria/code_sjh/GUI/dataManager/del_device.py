import csv

import numpy as np
import os


def del_data_all(dir,
                 err_key,
                 dst_dir = None):
	for file in os.listdir(dir):
		abs_file = os.path.join(dir, file)
		if not os.path.isfile(abs_file):
			continue
		data = np.loadtxt(abs_file, delimiter = ",")
		err = data == err_key
		err_ = err == False
		if np.sum(err_) > np.sum(err):
			continue
		if dst_dir is None:
			os.remove(abs_file)
		else:
			if not os.path.isdir(dst_dir):
				os.makedirs(dst_dir)
			os.replace(abs_file,os.path.join(dst_dir,file))

def del_data_work(src,err_key,dst = None):
	for dirs in os.listdir(src):
		abs_dir = os.path.join(src, dirs)
		if not os.path.isdir(abs_dir):
			continue
		del_data_all(abs_dir,err_key = err_key,dst_dir = dst)
def relabel(dir,labelfile = "labels.txt"):
	abs_labelfile = os.path.join(dir,labelfile)
	if os.path.isfile(abs_labelfile):
		os.remove(abs_labelfile)
	csvfiles = []
	csvlabels = []
	for dirs in os.listdir(dir):
		abs_dir = os.path.join(dir, dirs)
		if not os.path.isdir(abs_dir):
			continue
		for files in os.listdir(abs_dir):
			csvfiles.append(files)
			csvlabels.append(dirs)
	with open(abs_labelfile,"w") as f:
		csvwriter = csv.writer(f)
		csvwriter.writerow(csvfiles)
		csvwriter.writerow(csvlabels)



if __name__ == '__main__':
	del_data_work(os.path.join("class3ver3"),err_key = 0,dst = os.path.join("work_errdata_removed","err"))
	relabel("work_errdata_removed")