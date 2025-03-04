import os
import shutil
from bacteria.code_sjh.Core.basic_functions import getRootPath
import pandas
from av_pointwise import *
from split_pointwise import datafileSortbyPoint

coderoot = getRootPath("code_sjh")
projectroot = getRootPath("Raman_dl_ml_code")
data_root = os.path.join(os.path.dirname(projectroot),"data","脑胶质瘤")
dataroot_batch1 = os.path.join(
	data_root,r"原始数据\天坛神外 原始数据\天坛神外 光谱",
	"第一批")
dataroot_batch2 = os.path.join(
	data_root,r"原始数据\天坛神外 原始数据\天坛神外 光谱",
	"第二批 重新测试 2")
dataroot_batch3 = os.path.join(
	data_root,r"原始数据\天坛神外 原始数据\天坛神外 光谱",
	"第三批")
dataroot_batch4 = os.path.join(data_root,"原始数据",
	r"第四批\光谱")
dst_root = os.path.join(data_root, "data_reformed")


# batch1
def batch1():
	batch1_num2tissue = dict(
		病例2 = ["11"],
		病例5 = ["T1", "T3"],
		病例6 = ["H" + str(x) for x in range(1, 5)],
		病例8 = ["H" + str(x) for x in range(1, 6)],
		病例10 = ["T1", "T2"],
		病例12 = ["T" + str(x) for x in range(1, 4)],
		病例13 = None,
		病例15 = ["T" + str(x) for x in range(1, 3)],
		病例16 = ["T" + str(x) for x in range(1, 6)],
		病例26 = ["T" + str(x) for x in range(1, 6)],
		病例27 = ["T" + str(x) for x in range(1, 3)],
	)  # None代表all
	batch1_num2tissue["病例26"][3] = "T4MAPPING"
	keys = list(batch1_num2tissue.keys())

	T_paths = []  # "...\name\tissue"
	for dir_name in os.listdir(dataroot_batch1):
		dir_name_abs = os.path.join(dataroot_batch1, dir_name)
		if not os.path.isdir(dir_name_abs) or " " not in dir_name:
			continue
		name = os.path.basename(dir_name).split(" ")[0]
		if not name in keys:
			continue
		data_used = batch1_num2tissue[name]
		for dir_tissue in os.listdir(dir_name_abs):
			dir_tissue_abs = os.path.join(dir_name_abs, dir_tissue)
			if not os.path.isdir(dir_tissue_abs):
				continue
			if data_used is not None and dir_tissue not in data_used:
				continue

			T_paths.append(dir_tissue_abs)
			if data_used is not None:
				data_used.remove(dir_tissue)
		assert data_used is None or len(data_used) == 0, "{},{}".format(dir_name_abs, data_used)
	for path in T_paths:
		name, tissue = os.path.split(path)
		name = os.path.basename(name)
		if len(os.listdir(path)) == 0:
			continue
		shutil.copytree(path, os.path.join(dst_root, 'batch1', name.replace("病例", "p") + " " + tissue))


def count_file_dir(dir):
	"""
	return: num of files, num of dirs
	"""
	l = [os.path.join(dir, x) for x in os.listdir(dir)]
	a = len(l)
	file_num = len(list(filter(os.path.isfile, l)))
	return file_num, a - file_num


def batch2(dataroot_batch2):
	for dir_name in os.listdir(dataroot_batch2):
		dir_name_abs = os.path.join(dataroot_batch2, dir_name)
		infs = dir_name.split(" ")
		name = " ".join([infs[1].lower(), infs[0], infs[2]])
		if len(infs) == 4:
			dir_tissue = dir_name.split(" ")[-1]
			dst = os.path.join(dst_root, "batch2", name + " " + dir_tissue)
			if os.path.exists(dst):
				shutil.rmtree(dst)
			if count_file_dir(dir_name_abs)[0] > 0:
				shutil.copytree(dir_name_abs, dst)
			continue
		else:
			numfile, numdir = count_file_dir(dir_name_abs)
			if numdir == 0 and numfile != 0:
				dir_tissue = "T1"
				dst = os.path.join(dst_root, "batch2", name + " " + dir_tissue)
				if os.path.exists(dst):
					shutil.rmtree(dst)
				shutil.copytree(dir_name_abs, dst)
			elif numdir > 0:
				for dir_tissue in os.listdir(dir_name_abs):
					dir_tissue_abs = os.path.join(dir_name_abs, dir_tissue)
					if not os.path.isdir(dir_tissue_abs):
						continue
					if count_file_dir(dir_tissue_abs)[0] == 0:
						continue
					dst = os.path.join(dst_root, "batch2", name + " " + dir_tissue)
					if os.path.exists(dst):
						shutil.rmtree(dst)
					shutil.copytree(dir_tissue_abs, dst)


batch3_num = [
	60,
	122,
	110,
	127,
	74,
	91,
	109,
	49,
	136,
	83,
	113,
	130,
	61,
	97,
	120,
]


def batch3(dataroot_batch3):
	for dir_name in os.listdir(dataroot_batch3):
		dir_name_abs = os.path.join(dataroot_batch3, dir_name)
		infs = dir_name.split(" ")
		if not os.path.isdir(dir_name_abs):
			continue
		name = " ".join(["p" + infs[1], infs[0], *infs[2:]])
		num_p = int(infs[1])
		if not num_p in batch3_num:
			continue
		numfile, numdir = count_file_dir(dir_name_abs)
		if numdir == 0 and numfile != 0:
			dir_tissue = "T1"
			dst = os.path.join(dst_root, "batch3", name + " " + dir_tissue)
			if os.path.exists(dst):
				shutil.rmtree(dst)
			if count_file_dir(dir_name_abs)[0] > 0:
				shutil.copytree(dir_name_abs, dst)
		elif numdir > 0:
			for dir_tissue in os.listdir(dir_name_abs):
				dir_tissue_abs = os.path.join(dir_name_abs, dir_tissue)
				if not os.path.isdir(dir_tissue_abs):
					continue
				if count_file_dir(dir_tissue_abs)[0] == 0:
					continue
				dst = os.path.join(dst_root, "batch3", name + " " + dir_tissue)
				if os.path.exists(dst):
					shutil.rmtree(dst)
				shutil.copytree(dir_tissue_abs, dst)


def batch4(dataroot_batch4 = dataroot_batch4,
           dst_root = os.path.join(os.path.dirname(dst_root), "data_indep_unlabeled"),
           xs = None):
	if os.path.isdir(dst_root):
		shutil.rmtree(dst_root)
	os.makedirs(dst_root)
	readRaman = getRamanFromFile(0, 40000, dataname2idx = {"Wavelength": 2, "Intensity": 5})
	for person_dir in os.listdir(dataroot_batch4):
		person_dir_abs = os.path.join(dataroot_batch4, person_dir)
		info = person_dir.replace(",", " ")
		info = info.replace("，", " ")
		while "  " in info:
			info = info.replace("  ", " ")
		try:
			num, name, code = info.split(" ")
		except:
			continue

		for tissue_dir in os.listdir(person_dir_abs):
			tissue_dir_abs = os.path.join(person_dir_abs, tissue_dir)
			new_tissue_dir = " ".join([num, name, code, tissue_dir])
			new_tissue_dir_abs = os.path.join(dst_root, new_tissue_dir)
			shutil.copytree(tissue_dir_abs, new_tissue_dir_abs)
			for file in os.listdir(new_tissue_dir_abs):
				file_abs = os.path.join(new_tissue_dir_abs, file)
				refile(file_abs, readRaman, xs = xs)


def assert_times(dir, lasttime = None):
	times = []
	for file in os.listdir(dir):
		if not file.endswith(".csv"):
			continue
		time = file.split(".")[0].split(" ")[-1]
		time = [int(x) for x in time.split("_")]
		time = (time[0] * 60 + time[1]) * 60 + time[2]
		times.append(time)
	try:
		if max(times) - min(times) > 25:
			print(dir, " might have sth went wrong")
	except:
		print(times, ":", dir)
	if lasttime is not None and 0 < min(times) - lasttime < 10:
		print(dir, " datas of points got too close:{} -> {}".format(lasttime, min(times)))
	return max(times)


def count_files(dst_root):
	res = 0
	for r, d, f in os.walk(dst_root):
		res += len(f)
	return res




if __name__ == '__main__':
	# infofile = os.path.join(projectroot,"data","脑胶质瘤","data_used","第一二三批 病例编号&大类结果.xlsx")
	def get_infos(filename: str):
		df = pandas.read_excel(filename)
		nums = df["编号"]
		axes = df.axes[1]
		num2ele2label = {}

		for i in range(len(nums)):
			num2ele2label[nums[i]] = {}
			for j in range(4, 13):
				label_class: str = axes[j]
				label_class = label_class.replace("\\", "-")
				num2ele2label[nums[i]][label_class] = df[axes[j]][i]
		return num2ele2label


	# infos = get_infos(infofile)
	# print(infos)
	# ALL_root = os.path.join(dst_root,"all")
	# GBM_root = os.path.join(dst_root,"gbm")
	if os.path.exists(dst_root):
		shutil.rmtree(dst_root)
	batch1()
	batch2(dataroot_batch2)
	batch3(dataroot_batch3)
	dst_root_indep = os.path.join(os.path.dirname(dst_root), "data_indep_unlabeled")  # batch4用作完全独立验证集
	batch4(dataroot_batch4, dst_root = dst_root_indep)
	for i in range(3):
		batch = os.path.join(dst_root, "batch{}".format(i + 1))
		for dir in os.listdir(batch):
			datafileSortbyPoint(os.path.join(batch, dir), None if i == 0 else 3)

	classes = ["batch1", "batch2", "batch3"]
	# dir_prefix = "point"
	current_dir = os.path.dirname(__file__)

	from bacteria.code_sjh.bin.Glioma.data_handler.av_pointwise import *

	xs = None
	for c in classes:
		if c == "batch2":
			readRaman = getRamanFromFile(0, 40000, dataname2idx = {"Wavelength": 2, "Intensity": 5})
		elif c == "batch1":
			readRaman = getRamanFromFile(0, 40000, dataname2idx = {"Wavelength": 5, "Intensity": 4})
		else:
			readRaman = getRamanFromFile(0, 40000, dataname2idx = {"Wavelength": 2, "Intensity": 5})
		class_root = os.path.join(data_root, c)
		for dirs in os.listdir(class_root):
			abs_dir = os.path.join(class_root, dirs)
			if not os.path.isdir(abs_dir):
				continue
			for pointdir in os.listdir(abs_dir):
				abs_pointdir = os.path.join(abs_dir, pointdir)
				if not os.path.isdir(abs_pointdir):
					# or not pointdir.startswith(dir_prefix):
					continue
				dst_file = abs_pointdir + ".csv"
				for raman_file in os.listdir(abs_pointdir):
					abs_raman_file = os.path.join(abs_pointdir, raman_file)
					if xs is None:
						_, xs = readRaman(abs_raman_file)
					refile(abs_raman_file, readRaman, xs = xs)
				dir2file_av(abs_pointdir, readRaman, dst_file)
	# file2plot(dst_file, dst_file[:-4] + ".png", readRaman)

	classes = ["batch1", "batch2", "batch3"]
	db_root = os.path.join(os.path.dirname(data_root), "data")
	for c in classes:
		class_root = os.path.join(data_root, c)
		for tissue_dir in os.listdir(class_root):
			tissue_dir_abs = os.path.join(class_root, tissue_dir)
			shutil.copytree(tissue_dir_abs, os.path.join(db_root, "unlabeled", tissue_dir))

	for i in range(1, 3):
		batch = os.path.join(dst_root, "batch{}".format(i + 1))
		for tissue_dir in os.listdir(batch):
			tissue_dir_abs = os.path.join(batch, tissue_dir)
			last_time = None
			for point_dir in os.listdir(tissue_dir_abs):
				last_time = assert_times(os.path.join(tissue_dir_abs, point_dir), last_time)
	print(count_files(dst_root))

if __name__ == '__main__':
	classes = ["batch1", "batch2", "batch3"]
	# dir_prefix = "point"
	current_dir = os.path.dirname(__file__)

	xs = None
	for c in classes:
		if c == "batch2":
			readRaman = getRamanFromFile(0, 40000, dataname2idx = {"Wavelength": 2, "Intensity": 5})
		elif c == "batch1":
			readRaman = getRamanFromFile(0, 40000, dataname2idx = {"Wavelength": 5, "Intensity": 4})
		else:
			readRaman = getRamanFromFile(0, 40000, dataname2idx = {"Wavelength": 2, "Intensity": 5})
		class_root = os.path.join(data_root, c)
		for dirs in os.listdir(class_root):
			abs_dir = os.path.join(class_root, dirs)
			if not os.path.isdir(abs_dir):
				continue
			for pointdir in os.listdir(abs_dir):
				abs_pointdir = os.path.join(abs_dir, pointdir)
				if not os.path.isdir(abs_pointdir):
					# or not pointdir.startswith(dir_prefix):
					continue
				dst_file = abs_pointdir + ".csv"
				for raman_file in os.listdir(abs_pointdir):
					abs_raman_file = os.path.join(abs_pointdir, raman_file)
					if xs is None:
						_, xs = readRaman(abs_raman_file)
			# file2plot(dst_file, dst_file[:-4] + ".png", readRaman)
	dst_root_indep = os.path.join(os.path.dirname(dst_root), "data_indep_unlabeled", "unlabeled")  # batch4用作完全独立验证集
	batch4(dataroot_batch4, dst_root = dst_root_indep, xs = xs)