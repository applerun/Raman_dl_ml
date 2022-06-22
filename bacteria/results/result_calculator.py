import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from bacteria.code_sjh.utils.Validation.visdom_utils import data2mean_std


def cal_conf_matrix(file):
	cf = np.loadtxt(file, delimiter = ",")
	A, B, C, D = cf.flatten()
	acc = (A + D) / (A + B + C + D)
	sens = A / (A + B)
	spec = D / (C + D)
	return acc, sens, spec


def cam_show_mpl(file,
                 ax: plt.Axes = None,
                 container:list = None,
                 bias = 0.,
                 annotate_peak = True):
	if ax is None:
		fig, ax = plt.subplots()
	else:
		fig = ax.figure
	name = os.path.basename(file)[:-4]
	arr = np.loadtxt(file, delimiter = ",")
	xs = arr[0]
	cam = arr[1:]
	cam_mean = cam.mean(axis = 0)+bias
	cam_std = cam.std(axis = 0)
	cam_up, cam_down = cam_mean + cam_std, cam_mean - cam_std

	ax.plot(xs, cam_mean, color = "red")
	ax.fill_between(xs, cam_up, cam_down, color = "pink")
	x_p = xs[0]
	count = 0
	ax.set_title(name)
	ax.set_xlabel("wavenumber cm$^{-1}$")
	ax.set_ylabel("intensity")
	if not annotate_peak:
		return fig

	peaks_i, properties = signal.find_peaks(cam_mean, distance = 5, prominence = (cam_mean.mean()-bias, 1))
	for i in range(len(peaks_i)):

		p_m = properties["prominences"].mean()
		pi = peaks_i[i]
		prominence = properties["prominences"][i]
		# if cam_std[i]>cam_std.mean()/4 and cam_std[i]>0.1:
		# 	continue
		# if prominence <p_m:
		# 	continue
		# txt = "wavenum:{:.2f}\nprom:{:.2f}".format(xs[pi],prominence)
		txt = "{:.1f}".format(xs[pi])
		# 避免文字重叠
		y = xs[pi] / (xs[-1] - xs[0]) * 0.7
		y_t = y + (1 - y) * cam_up[pi]
		y_c = y_t - cam_up[pi]
		if xs[pi] - x_p < 100:
			count += 1
			y_t -= min(y_c, count / 3 * y_c)
		else:
			count = 0
			x_p = xs[pi]
		if not container is None:
			container.append([xs[pi],prominence])
		ax.annotate(txt, xy = (xs[pi], cam_up[pi]), xytext = (xs[pi], y_t), fontsize = 6,
		            arrowprops = dict(arrowstyle = "->") if y_c > 0.1 else None,
		            horizontalalignment = 'center'
		            )


	return fig

def neck_vis(dir,label2name:dict,ax:plt.Axes = None):
	if ax is None:
		fig, ax = plt.subplots()
	else:
		fig = ax.figure
	for label in label2name.keys():
		name = label2name[label]
		file = name+"_VAE_neck.csv"
		file_p = os.path.join(dir,file)
		h = np.loadtxt(file_p,delimiter = ",")
		ax.scatter(h[:, 0], h[:, 1], label = name)


def main_CAM():
	dir = os.path.join("liver", "bals", "alexnet")
	label2name = {0: "HepG2", 1: "MIHA"}
	container = []
	biases = [0, 0]
	fig_all, axes = plt.subplots(1, 2)
	for d in os.listdir(dir):
		d_p = os.path.join(dir, d)
		if not d.startswith("n") or not os.path.isdir(d_p):
			continue
		cam_dir = os.path.join(d_p, "cam")
		for file in os.listdir(cam_dir):
			file_p = os.path.join(cam_dir, file)
			if not file.endswith(".csv") or not os.path.isfile(file_p):
				continue
			if file.startswith("test"):
				continue
			for label in label2name.keys():
				name = label2name[label]
				if name in file:
					cam_show_mpl(file_p, axes[label], bias = biases[label], annotate_peak = False)
					biases[label] += 1
			fig = cam_show_mpl(file_p, container = container)
			plt.figure(fig)
			plt.show()
			if not os.path.isdir(os.path.join(d_p, "cam_plot")):
				os.makedirs(os.path.join(d_p, "cam_plot"))
			fig.savefig(os.path.join(d_p, "cam_plot", file[:-4] + ".png"))
			plt.close(fig)
	plt.figure(fig_all)
	plt.show()
	assert biases[0] == biases[1] == 6, biases

	f = open("all_peaks.csv", "w", newline = "")
	writer = csv.writer(f)
	writer.writerow(["Peaks", "Prominences"])
	c = sorted(container, key = lambda x: x[0])
	w = 0
	for l in c:
		l_p = l
		p = [l[1]]
		if abs(l[0] - w) < 1e-2:
			p.append(l[1])
			continue
		# else:
		# 	p = []
		# writer.writerow([w,sum(p)/len(p)])
		writer.writerow(l)
		w = l[0]
	f.close()
def main_CVAE_neckvis():
	dir_r = os.path.join("liver_CVAE", "Record_2022-05-01-08_58_35")
	label2name = {0: "HepG2", 1: "MIHA"}
	for dir in os.listdir(dir_r):
		dir = os.path.join(dir_r,dir)
		if not os.path.isdir(dir):
			continue
		for d in os.listdir(dir):
			d_p = os.path.join(dir, d)
			if not d.startswith("n") or not os.path.isdir(d_p):
				continue
			for record_dirname in ["val_record","test_record"]:
				dbname =record_dirname.split("_")[0]
				neck_dir = os.path.join(d_p, record_dirname)
				fig, ax = plt.subplots()
				typs = []
				names =[]
				for file in os.listdir(neck_dir):
					file_p = os.path.join(neck_dir, file)
					if not file.endswith(".csv") or not os.path.isfile(file_p):
						continue
					name = file.split("_")[0]
					h = np.loadtxt(file_p,delimiter = ",")

					typs.append(ax.scatter(h[:, 0], h[:, 1], label = name))
					names.append(name)
				plt.legend(tuple(typs), tuple(names))
				plt.figure(fig)
				plt.show()

				fig.savefig(os.path.join(d_p, dbname + "_neck_vis.png"))

				plt.close(fig)
def get_res_matrix(dir,dst):
	if not os.path.isdir(os.path.dirname(dst)):
		os.makedirs(os.path.dirname(dst))
	l = []
	for csvfile in os.listdir(dir):
		if not csvfile.endswith(".csv"):
			continue
		csvfile_p = os.path.join(dir,csvfile)
		with open(csvfile_p) as f:
			lines = list(f.readlines())
			line = lines[1]
			line = line.replace(r", 'viz':","")
			line = line.replace("'criteon': BCELoss(),","")
			line = re.sub("device\(.*?\)","None",line)
			res = re.sub("\<.*?\>", '', line)
			dict = eval(res)
			rates = dict["rates"]
			label_rate,kld_rate = rates["label_rate"],rates["kld_rate"]
			acces = []
			for l_t in lines:
				if not l_t.startswith("0"):
					continue
				acces.append(float(l_t.split(",")[2]))
			# acc = float(lines[-1].split(",")[2].split(" +- ")[0])
			acc = np.mean(acces)
			l.append([label_rate,kld_rate,acc])
	l = np.array(l)
	np.savetxt(dst,l,delimiter = ",",header = "label_rate,kld_rate,acc")

def main_CVAE_rates():
	dir_r = os.path.join("liver_CVAE", "Record_2022-05-01-08_58_35")
	dst = os.path.join("liver_CVAE", "Record_2022-05-01-08_58_35.csv")
	get_res_matrix(dir_r,dst)
if __name__ == '__main__':
	# rdir = os.path.join("liver", "CVAE")
	# rdir = os.path.join("liver_CVAE","Record_2022-05-01-08_58_35")
	rdir = os.path.join("radar/Record_2022-06-17-16_41_42/RecordFri_Jun_17_16-41-42_2022")
	for dir in os.listdir(rdir):
		dir = os.path.join(rdir, dir)
		if not os.path.isdir(dir):
			continue
		acc, sens, spec = cal_conf_matrix(os.path.join(dir, "conf_matrix","test_confusion_matrix.csv"))
		print("{}:accuracy-{},sensitivity-{},specificity-{}".format(dir, acc, sens, spec), )

	# main_CVAE_rates()



