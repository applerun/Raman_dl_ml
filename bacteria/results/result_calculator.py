import os

import numpy as np
def cal_conf_matrix(file):
	cf = np.loadtxt(file,delimiter = ",")
	A,B,C,D = cf.flatten()
	acc = (A+D)/(A+B+C+D)
	sens = A/(A+B)
	spec = D/(C+D)
	return acc,sens,spec
if __name__ == '__main__':
	rdir = os.path.join("liver","brnp")

	for dir in os.listdir(rdir):
		dir = os.path.join(rdir,dir)
		if not os.path.isdir(dir):
			continue
		acc,sens,spec = cal_conf_matrix(os.path.join(dir,"test_confusion_matrix.csv"))
		print("{}:accuracy-{},sensitivity-{},specificity-{}".format(dir,acc,sens,spec),)