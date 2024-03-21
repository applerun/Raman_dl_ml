import csv
import os
import shutil, warnings

import numpy

from bacteria.code_sjh.utils.RamanData import getRamanFromFile, Raman_depth_gen, data_leak_check_by_filename, \
	get_dict_str, Raman_dirwise

readdatafunc = getRamanFromFile(
	# 定义读取数据的函数
	wavelengthstart = 39, wavelengthend = 1810, delimeter = None, dataname2idx = {"Wavelength": 2, "Intensity": 6}
)


def cal_spec(src, dst, datasplit = "personwise"):
	dataroot_ = src
	personwise = datasplit == "personwise"
	if not os.path.isdir(dst):
		os.makedirs(dst)
	if personwise:
		from bacteria.code_sjh.bin.Glioma.data_handler.samplewise2personwise import rename_files_between
		dataroot_dst = dataroot_ + "_renamed_for_personwise"
		# rename_files_between(dataroot_dst, 3)
		if not os.path.isdir(dataroot_dst):
			if os.path.isdir(dataroot_dst + "_failed"):
				shutil.move(dataroot_dst + "_failed", dataroot_dst)
			else:
				shutil.copytree(dataroot_, dataroot_dst)
			try:
				rename_files_between(dataroot_dst, 3)
			except:
				warnings.warn("rename_failed")
				shutil.move(dataroot_dst, dataroot_dst + "_failed")
				dataroot_dst = dataroot_
	else:
		dataroot_dst = dataroot_
	for mole in os.listdir(dataroot_dst):
		mole_abs = os.path.join(dataroot_dst, mole)
		if not os.path.isdir(dataroot_):
			continue
		raman = Raman_depth_gen(2, 2) if datasplit == "pointwise" else Raman_dirwise
		db_cfg = dict(  # 数据集设置
			dataroot = mole_abs,
			backEnd = ".csv",
			# backEnd = ".asc",
			mode = "all",
			LoadCsvFile = readdatafunc,
			transform = None, shuffle = False)
		sfpath = "{}_stat.csv".format(datasplit)
		all_db = raman(**db_cfg, sfpath = sfpath)
		dst_stat_file = os.path.join(dst, mole + "_" + sfpath)
		if os.path.isfile(dst_stat_file):
			os.remove(dst_stat_file)
		os.rename(os.path.join(mole_abs, all_db.sfpath), dst_stat_file)


def files2numfiles(src, dst):
	if not isinstance(src, list):
		src = [src]
	dst_f = open(dst, "w", newline = "")
	header = ["person", "num_points", ]
	persons = None
	persons2numfiles = None
	new_lines = []
	numfileses = None
	dst_writer = csv.writer(dst_f)
	sum_line = ["sum"]
	sum_line_2 = ["ratio",""]
	for src_file_path in src:
		src_f = open(src_file_path, "r")

		src_reader = csv.reader(src_f)

		lines = list(src_reader.__iter__())
		persons_ = []
		numfiles_ = []
		labels_ = []
		for i, line in enumerate(lines):
			if i == 0:
				continue
			label = line[0]
			labels_.append(label)
			person = os.path.basename(line[1])
			persons_.append(person)
			numfiles = len(line) - 2
			numfiles_.append(numfiles)
		src_f.close()
		mole = os.path.basename(src_file_path).split("_")[0]

		header.append(mole)
		if persons2numfiles is None:
			persons = persons_
			numfileses = numfiles_
			persons2numfiles = dict(zip(persons, numfileses))
			new_lines = [[persons_[i], numfiles_[i], labels_[i]] for i in range(len(persons_))]
			sum_line.append(sum(numfileses))
		else:
			persons_2numfiles_ = dict(zip(persons_, numfiles_))
			for p, f in persons_2numfiles_.items():
				assert persons2numfiles[p] == f
				new_lines[persons.index(p)].append(labels_[persons_.index(p)])
		num_pos = (numpy.array(labels_, dtype = int) * numpy.array(numfiles_, dtype = int)).sum()
		sum_line.append(num_pos)
		sum_line_2.append("{:.2f}%".format(num_pos/sum_line[1]*100))
	dst_writer.writerows([header] + new_lines+[sum_line,sum_line_2])

	dst_f.close()


if __name__ == '__main__':
	from bacteria.code_sjh.Core.basic_functions.path_func import getRootPath

	projectroot = getRootPath("Raman_dl_ml")
	glioma_data_root = os.path.join(projectroot, "data", "脑胶质瘤")

	dataroot_ = os.path.join(glioma_data_root, r"labeled_data\data_indep_labeled")
	res_stat_root = os.path.join(glioma_data_root, r"data_stat\data_indep_labeled")
	# cal_spec(dataroot_, res_stat_root)
	src_files = [os.path.join(res_stat_root, f) for f in os.listdir(res_stat_root)]
	files2numfiles(src_files, res_stat_root + ".csv")
