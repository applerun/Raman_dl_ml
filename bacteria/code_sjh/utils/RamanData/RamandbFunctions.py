import os.path
import warnings
from bacteria.code_sjh.Core.RamanData import RamanDatasetCore
from bacteria.code_sjh.Core.basic_functions import mpl_func
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba_array, to_rgba


def save_csv_file_info(Raman: RamanDatasetCore,
                       dst):
	"""
	TODO:根据Raman类型在dst处生成对应的信息文件
	@param Raman:
	@param dst:
	@return:
	"""
	if Raman.mode is not "all":
		warnings.warn("The raman db mode is not 'all':{}, Please Check".format(Raman.mode))
	if not os.path.isdir(os.path.dirname(dst)):
		os.makedirs(os.path.dirname(dst))
	label2name = Raman.label2name()
	with open(dst, "w") as f:
		for i in range(len(Raman)):
			label = Raman[i]
			label = int(label)
			label = str(label)
			file = Raman.RamanFiles[i]


def data_leak_check_by_filename(dbs):
	"""
	输入一系列RamanDatasetCore的子集，返回同时在至少两个db中存在的文件
	"""
	if len(dbs) is 2:
		file_list1 = dbs[0].RamanFiles
		file_list2 = dbs[1].RamanFiles
		leaked_data_file = list(set(file_list1) & set(file_list2))
	else:
		leaked_data_file = []
		for i in range(len(dbs)):
			for j in range(i, len(dbs)):
				leaked_data_file += data_leak_check_by_filename([dbs[i], dbs[j]])
	return leaked_data_file


def db_plot(db: RamanDatasetCore,
            # svdir = os.path.join(project_root, "Sample_results", "Sample_bad_signal", "plot"),
            newX = None,
            # 是否要插值
            pltsave_in_one = False,
            # 是否保存所有处理后结果在一张图中
            dir_wise = False,
            # True :每个tissue 单独画图，labelwise ：每个label单独画图
            axes: plt.Axes or list = None,
            bias = 0,
            min_bias_interv = 0,
            ):
	"""
	database数据可视化
	@param db: 数据库，RamanDatasetCore
	@param newX: None 所有光谱的横坐标，默认从db提取
	@param pltsave_in_one: 是否将所有的数据绘制在同一张图上，若是，将自动设置每个光谱的bias，保证光谱不重叠，自动设置legend（设置在axes外）
	@param dir_wise: 仅对Raman_dirwise生效，是否每个tissue单独画图
	@param axes: None，指定光谱绘制的Axes，若绘制在不同的Axes上，需要为Axes的列表
	@param bias: 将所有曲线向上平移bias
	@param min_bias_interv: 仅在pltsave_in_one=True时生效，设置不同光谱相对距离最小值
	@return:
	"""
	colors = list(mcolors.XKCD_COLORS.keys())
	if dir_wise:
		name2data_tissue = db.get_data_sorted_by_sample()  # 按照子文件夹画图
	else:
		name2data_tissue = db.get_data_sorted_by_name()  # 选择分类画图
	if pltsave_in_one:
		if axes is None:
			fig, ax = plt.subplots(1, 1)
		else:
			fig = axes.figure
			ax = axes

	else:
		assert axes is None or type(axes) is list
		if type(axes) is list:
			assert len(axes) == db.numclasses
		fig = None
	figs = []
	c = mcolors.XKCD_COLORS[colors[0]]
	for name in name2data_tissue.keys():
		tissue_data = name2data_tissue[name].numpy()
		if pltsave_in_one:
			c = mcolors.XKCD_COLORS[colors.pop()]
			mpl_func.spectrum_vis_mpl(tissue_data, db.xs if newX is None else newX, name = name, ax = ax,
			                          bias = bias,
			                          title = "processed",
			                          line_color = to_rgba(c, 1), shadow_color = to_rgba(c, 0.6)
			                          )
			bias += max(tissue_data.max(), min_bias_interv)
		else:
			fig, ax = plt.subplots(1, 1) if axes is None else axes.pop()

			mpl_func.spectrum_vis_mpl(tissue_data, db.xs if newX is None else newX, name = name, ax = ax,
			                          bias = bias,
			                          title = "processed",
			                          line_color = to_rgba(c, 1), shadow_color = to_rgba(c, 0.6)
			                          )
			bias += max(tissue_data.max(), min_bias_interv)
			figs.append(fig)
			plt.close(fig)
	if pltsave_in_one:
		ax.legend(loc = 'upper left', bbox_to_anchor = (1.05, 1))
		return fig
	else:
		return figs
