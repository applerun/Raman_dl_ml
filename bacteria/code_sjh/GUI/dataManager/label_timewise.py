import copy
import csv
import os
import sys

import numpy
import pyqtgraph

coderoot = os.path.split(os.path.split(os.path.split(__file__)[0])[0])[0]
projectroot = os.path.split(coderoot)[0]
from bacteria.code_sjh.utils.Process_utils.errhandler import all_eval_err_handle

try:
	from bacteria.code_sjh.GUI.dataManager.Windows.labelize_minwise import *
	from bacteria.code_sjh.GUI.dataManager.Widgets import labelize_wigdets
except:
	from bacteria.code_sjh.GUI.dataManager.Windows.labelize_minwise import *
	import bacteria.code_sjh.GUI.dataManager.Widgets.labelize_wigdets
from bacteria.code_sjh.GUI.dataManager.Widgets.labelize_wigdets_func import Label_Widget
from PyQt5.Qt import *
from shutil import copyfile


def getInfoFromFilename(filename):
	filename = os.path.split(filename)[1]
	Infos = filename.split("_")
	device, date = Infos
	return device, date


import pyqtgraph as pg

pg.setConfigOption('background', 'w')


# from scipy.signal import savgol_filter
#
#
# def sg_filter(window_length = 15, polyorder = 3):
#     def func(x):
#         x = savgol_filter(x, window_length, polyorder)
#         return x
#
#     return func
def night(data):
	data = data[900:1260]
	return data


class timeStringAxis(pg.AxisItem):
	def __init__(self,
	             xs,
	             strs,
	             *args,
	             **kwargs):
		pg.AxisItem.__init__(self, *args, **kwargs)
		self.x_values = xs
		self.x_strings = strs

	def tickStrings(self,
	                values,
	                scale,
	                spacing):
		strings = []
		for v in values:
			# vs is the original tick value
			vs = v * scale
			# if we have vs in our values, show the string
			# otherwise show nothing
			if vs in self.x_values:
				# Find the string with x_values closest to vs
				vstr = self.x_strings[numpy.abs(self.x_values - vs).argmin()]
			else:
				vstr = ""
			strings.append(vstr)
		return strings


def mean_func_nozero(data):
	t = 0
	t_l = 0
	for i in data:
		if i == 0:
			continue
		t += i
		t_l += 1
	return 0 if t_l == 0 else t / t_l


def norm_transform(data):
	data = night(data)  # 取晚上的数据
	res = all_eval_err_handle(data)
	# res = [mean_func_nozero(data[max(i - 5, 0):min(i + 5, len(data))]) for i in range(len(data))]  # 均值
	# res = [mean_func_nozero(data[max(i - 3, 0):min(i + 3, len(data))]) for i in range(len(data))]  # 均值
	return res


class LabelWindow(QMainWindow, Ui_MainWindow):
	sigLabelAdded = QtCore.pyqtSignal(object)
	sigDataChanged = QtCore.pyqtSignal(object)
	sigDataChanging = QtCore.pyqtSignal(object)
	def __init__(self,
	             parent = None,
	             transform = None,
	             starttime = 0):
		self.starttime = starttime
		super(LabelWindow, self).__init__(parent)
		self.setupUi(self)
		self.action_history = []
		self.transform = transform
		self.initButton()
		self.initEnv()

	def initEnv(self):
		# 添加辅助线
		self.hLine_low = pg.InfiniteLine(angle = 0, movable = False,
		                                 pen = pyqtgraph.mkPen(width = 2, dash = (1.8, 1.8)))
		self.hLine_high = pg.InfiniteLine(angle = 0, movable = False,
		                                  pen = pyqtgraph.mkPen(width = 2, dash = (1.8, 1.8)))
		self.hLine_idx = pg.InfiniteLine(angle = 0, movable = False,
		                                 pen = pyqtgraph.mkPen(width = 2, dash = (1.8, 1.8)))
		self.hLine_idx.setPos(0.65)
		self.vLine = pg.InfiniteLine(angle = 90, movable = False)
		self.hLine = pg.InfiniteLine(angle = 0, movable = False)

		# 数据显示范围
		self.region = pg.LinearRegionItem()

		self.region.sigRegionChanged.connect(self.update_p1range)
		self.data_plot.sigRangeChanged.connect(self.update_region)

		self.showcrosshair = True
		self.cam_show = False
		self.proxy = pg.SignalProxy(self.data_plot.scene().sigMouseMoved, rateLimit = 60, slot = self.mouseMoved)
		# self.data_plot.scene().sigMouseMoved.connect(self.mouseMoved)
		self.data_sequence = ["breath", "heart", "idx"]
		self.workpath = os.path.join(os.path.dirname(__file__), "cache", "work")
		self.datapath = os.path.join(os.path.dirname(__file__), "cache", "data")
		self.label_conf_file = os.path.join(os.path.dirname(__file__), "cache", "labelfile.csv")
		self.data = numpy.zeros((3, 1553))
		self.data_show_region = None
		self.label2color = {"Norm": QtGui.QColor(255,255,255,255)}
		self.timeaxis = False
		self._currentFile = None
		self._currentIdx = 0
		self._currentData = 1
		self.files = []
		self.Infos = []
		self.initdataBase()
		self.initWorkPlace()
		self.load_label()
		self.changeTo(0)
		self.changeData(0)
		self.input_data_path.setText(self.datapath)
		self.input_work_path.setText(self.workpath)

	def initButton(self):
		self.set_data_path.clicked.connect(self.set_data_path_clicked)
		self.set_work_path.clicked.connect(self.set_work_path_clicked)
		self.redo_b.clicked.connect(self.redo)
		self.undo_b.clicked.connect(self.undo)
		self.next_b.clicked.connect(self.next_data)
		self.prev_b.clicked.connect(self.prev_data)
		self.nu_b.clicked.connect(self.next_unclassified_data)
		self.fu_b.clicked.connect(self.first_unclassified_data)
		self.name_list.currentRowChanged.connect(self.changeData)
		self.cam_b.clicked.connect(self.cam_button_clicked)
		self.data_list.itemClicked.connect(self.itemClicked)
		self.refresh_b.clicked.connect(self.refresh)

	def keyPressEvent(self,
	                  event: QtGui.QKeyEvent) -> None:

		if QApplication.keyboardModifiers() == Qt.ControlModifier:
			if event.key() == Qt.Key_Z:
				self.undo()
			elif event.key() == Qt.Key_Y:
				self.redo()
			elif event.key() == Qt.Key_W:
				self.close()
		elif QApplication.keyboardModifiers() == Qt.ShiftModifier:
			if event.key() == Qt.Key_A:
				self.prev_data()
			elif event.key() == Qt.Key_D:
				self.next_data()
			elif event.key() == Qt.Key_W:
				self.norm_c()
			elif event.key() == Qt.Key_S:
				self.abnorm_c()
		else:
			if event.key() == Qt.Key_Equal:
				self.next_unclassified_data()
			elif event.key() == Qt.Key_Minus:
				self.first_unclassified_data()
			elif event.key() == Qt.Key_Delete:
				self.abandon()

	def itemClicked(self,
	                item: QListWidgetItem):
		self.changeTo(self.files.index(item.text()))
		self.action_info.setText("currrentFile-" + self._currentFile)

	def initdataBase(self, ):
		self.files = []
		self.Infos = []
		self.data_list.clear()
		if not os.path.isdir(self.datapath):
			os.makedirs(self.datapath)
		for file in os.listdir(self.datapath):
			f = os.path.join(self.datapath, file)
			if not os.path.isfile(f) or not f.endswith(".csv"):
				continue
			self.files.append(file)
			device, date = getInfoFromFilename(os.path.join(self.datapath, file))
			self.Infos.append(dict(
				device = device,
				date = date[:-4],
				datafilename = file
			))
			self.data_list.addItem(QListWidgetItem(file))

	def changeData(self,
	               idx):
		self._currentData = idx
		self.data_label.setText(self.data_sequence[self._currentData])
		self.showdata()
		self.sigDataChanged.emit(self)

	def initWorkPlace(self, ):
		self.label2name = ["Norm", "Abnorm", "Abandoned"]
		self.class_num = len(self.label2name)
		# 准备分类后的文件夹
		for l in range(self.class_num):
			if not os.path.exists(os.path.join(self.workpath, self.label2name[l])):
				os.makedirs(os.path.join(self.workpath, self.label2name[l]))
		# 创建缓存空间
		if not os.path.isdir(os.path.join(os.path.dirname(__file__), "cache")):
			os.makedirs(os.path.join(os.path.dirname(__file__), "cache"))
		return

	def savelabel(self,
	              file):
		return

	def savelabelmask(self,
	                  data,
	                  file):
		return

	def load_label(self,
	               filename = "labels.txt"):
		self.labelfile = filename
		if not os.path.exists(os.path.join(self.workpath, filename)):
			self.labels = ["unclassified"] * len(self)
			self.save_label()
		else:
			with open(os.path.join(self.workpath, filename), "r", newline = "") as f:
				self.labels = [x for x in f.readlines()[1]]
			if len(self.labels) < len(self):
				os.remove(os.path.join(self.workpath, filename))
				self.load_label(filename)
		for datafilename in self.files:
			for l in self.label2name:
				if datafilename in os.listdir(os.path.join(self.workpath, l)):
					idx = self.files.index(datafilename)
					self.labels[idx] = l

	def save_label(self,
	               filename = None):
		if filename == None:
			filename = os.path.join(self.workpath, self.labelfile)
		with open(filename, "w") as f:
			writer = csv.writer(f)
			writer.writerow(self.files)
			writer.writerow(self.labels)

	def __len__(self):
		return len(self.files)

	def refresh(self):
		self.showdata()
		self.showInfo()

	def set_data_path_clicked(self, ):
		self.changeTo(0)
		data_path = QtWidgets.QFileDialog.getExistingDirectory(self, "浏览",
		                                                       os.path.join(os.path.dirname(__file__), "data"))
		self.datapath = data_path
		self.input_data_path.setText("data_path: " + data_path)
		self.initdataBase()
		self.load_label()
		self.changeTo(0)

		return

	def set_work_path_clicked(self, ):
		data_path = QtWidgets.QFileDialog.getExistingDirectory(self, "浏览",
		                                                       os.path.join(os.path.dirname(__file__), "cache", "work"))
		self.datapath = data_path
		self.input_work_path.setText("work_path: " + data_path)
		self.initWorkPlace()
		self.load_label()
		self.changeTo(0)
		return

	def cam_button_clicked(self):
		if self.cam_show:
			self.cam_show = False
			self.refresh()
			return
		else:
			self.cam_show = True
			self.cam_path = QtWidgets.QFileDialog.getExistingDirectory(self, "浏览",
			                                                           os.path.join(os.path.dirname(__file__), "data"))
		return

	def mouseMoved(self,
	               evt):
		pos = evt[0]  ## using signal proxy turns original arguments into a tuple
		if self.data_plot.sceneBoundingRect().contains(pos):
			vb = self.data_plot.centralWidget.vb
			mousePoint = vb.mapSceneToView(pos)
			index = int(mousePoint.x())
			if index > -2 and index < len(self.data) // 60 + 10:
				time = float(mousePoint.x()) + self.starttime
				time = time % 24
				h = int(time)
				m = (time - h) * 60
				time = "{}:{:0>2}".format(h, int(m))
				self.poslabel.setText(str(time))
				self.vLine.setPos(mousePoint.x())
				self.hLine.setPos(mousePoint.y())

	def changeTo(self,
	             idx):
		self.sigDataChanging.emit(self)
		if len(self.files) == 0:
			return
		if self.labels[self._currentIdx] == "unclassified":
			self.data_list.item(self._currentIdx).setBackground(QColor('white'))
		elif self.labels[self._currentIdx] == "Abandoned":
			self.data_list.item(self._currentIdx).setBackground(QColor('pink'))
		elif self.labels[self._currentIdx] in ["Norm", "Abnorm"]:
			self.data_list.item(self._currentIdx).setBackground(QColor('yellow'))
		self._currentIdx = idx
		self._currentFile = self.files[self._currentIdx]
		self.data_list.item(self._currentIdx).setBackground(QColor('blue'))
		self.refresh()
		self.sigDataChanged.emit(self)

	def readdatafile(self,
	                 filepath):
		with open(filepath) as f:
			data = [line for line in csv.reader(f)][self._currentData]
		self.data = numpy.array([float(x) for x in data])
		if self.transform != None:
			self.data = self.transform(self.data)

	def showInfo(self):
		self.data_info.setText(
			"""数据信息：\n\t第{}/{}份数据。\n\t设备号：{}；\n\t日期：{}；\n\t状态：{}""".format(
				self._currentIdx + 1,
				len(self),
				self.Infos[self._currentIdx]["device"],
				self.Infos[self._currentIdx]["date"],
				self.labels[self._currentIdx]
			)

		)
		return

	def undo(self):
		self.action_info.setText("undo")
		return

	def redo(self):
		return

	def prev_data(self):
		self.action_info.setText("Prev")
		new_idx = (self.__len__() + self._currentIdx - 1) % self.__len__()
		self.changeTo(new_idx)
		return

	def next_data(self):
		self.action_info.setText("Next")
		new_idx = (self._currentIdx + 1) % self.__len__()
		self.changeTo(new_idx)
		return

	def next_unclassified_data(self):
		self.action_info.setText("next_unclassified_data")
		for i in range(self._currentIdx, len(self)):
			self.changeTo(i)
			if self.labels[i] == "unclassified":
				# for a in range(self._currentIdx, i):
				#     self.data_list.item(a).setBackground(QColor('yellow'))
				# self.chage(i)
				return

	def first_unclassified_data(self):
		self.action_info.setText("first_unclassified_data")
		for i in range(self._currentIdx + 1):
			self.changeTo(i)
			if self.labels[i] == "unclassified":
				# for a in range(0, i):
				#     self.data_list.item(a).setBackground(QColor('yellow'))
				# self.chage(i)
				return
		return

	def norm_c(self):  # TODO:将当前文件复制至
		self.action_info.setText("norm")
		if self.labels[self._currentIdx] == "Norm":
			return
		if self.labels[self._currentIdx] != "unclassified":
			os.remove(os.path.join(self.workpath, self.labels[self._currentIdx], self._currentFile))

		self.labels[self._currentIdx] = "Norm"
		copyfile(os.path.join(self.datapath, self._currentFile, ),
		         os.path.join(self.workpath, "Norm", self._currentFile))
		self.save_label()
		self.next_data()
		return

	def abnorm_c(self):
		self.action_info.setText("abnorm")
		if self.labels[self._currentIdx] == "Abnorm":
			return
		if self.labels[self._currentIdx] != "unclassified":
			os.remove(os.path.join(self.workpath, self.labels[self._currentIdx], self._currentFile))
		self.labels[self._currentIdx] = "Abnorm"
		if not self._currentFile in os.listdir(os.path.join(self.workpath, "Abnorm")):
			copyfile(os.path.join(self.datapath, self._currentFile),
			         os.path.join(self.workpath, "Abnorm", self._currentFile))
		self.save_label()
		self.next_data()
		return

	def abandon(self):
		self.action_info.setText("abandoned")
		if self.labels[self._currentIdx] == "Abandoned":
			return
		if self.labels[self._currentIdx] != "unclassified":
			os.remove(os.path.join(self.workpath, self.labels[self._currentIdx], self._currentFile))

		self.labels[self._currentIdx] = "Abandoned"
		copyfile(os.path.join(self.datapath, self._currentFile),
		         os.path.join(self.workpath, "Abandoned", self._currentFile))
		self.save_label()
		self.next_data()
		return


class LabelWindow_colored(LabelWindow):
	def __init__(self,
	             parent = None,
	             transform = None,
	             starttime = 0):
		self.sup_1 = [8., 50., 0.15]
		self.sup_2 = [26, 85, 1.]
		super(LabelWindow_colored, self).__init__(parent, transform, starttime)

	def showdata(self):

		if len(self.files) == 0:
			return
		filepath = os.path.join(self.datapath, self._currentFile)
		self.timeSelectPlot.clear()
		self.data_plot.clear()
		self.readdatafile(filepath)
		self.set_ranges()
		xs = numpy.linspace(0, round(len(self.data) / 60), len(self.data))
		if self.data_show_region is None:
			self.data_show_region = [0, xs[-1]]
		# timeaxis = timeStringAxis(xs=xs,strs=(22+xs)%24,orientation = "bottom")
		self.data_plot_item = self.data_plot.plot(xs, self.data,
		                          pen = pg.mkPen(width = 4, dash = (1, 2)),
		                          symbolBrush = self.get_brushes(),
		                          symbolPen = self.get_pens(),
		                          symbol = "s",
		                          symbolSize = 3)
		p2d = self.timeSelectPlot.plot(xs, self.data)

		# 创建选取Item
		if self.region is None:
			self.region = pyqtgraph.LinearRegionItem()
		self.timeSelectPlot.addItem(self.region, ignoreBounds = True)
		self.region.setClipItem(p2d)
		self.region.setRegion(self.data_show_region)
		if self.cam_show:
			camfile = os.path.join(self.cam_path, self._currentFile[:-4] + ".cam.csv")
			if os.path.exists(camfile):
				cam_data = numpy.loadtxt(camfile, delimiter = ",")
				self.data_plot.plot(
					numpy.linspace(0, round(len(self.data) / 60), len(cam_data)),
					cam_data * self.data.max(),
					pen = pg.mkPen("r")
				)
		# self.data_plot.plot(xs, self.data, clear = True, pen = pg.mkPen(color = "363be1", width = 4, dash = (1, 2)))
		self.show_support_line()
		if not self.timeaxis:
			self.set_time_axis()
			self.set_time_axis(self.timeSelectPlot)
			self.timeaxis = True
		return

	def update_region(self,
	                  window,
	                  viewRange):
		rgn = viewRange[0]
		self.region.setRegion(rgn)

	def update_p1range(self):
		self.region.setZValue(10)
		minx, maxx = self.region.getRegion()
		self.data_plot.setXRange(minx, maxx, padding = 0)
		self.data_show_region = [minx, maxx]

	def its2col(self,
	            intensity):
		sup1 = self.sup_1[self._currentData]
		sup2 = self.sup_2[self._currentData]
		stat2col = {0: "#363be1", 1: "#5bc1e2", 2: "#f00"}
		stat = int(intensity > sup1) + int(intensity > sup2)
		col = stat2col[stat]
		if self._currentData == 2 and 0.65 < intensity <= 1:
			col = "#c0ca33"
		return col

	def get_brushes(self):
		# color_for_points = [self.its2col(x) for x in self.data]

		# brushes = [QtGui.QBrush(QtGui.QColor(col)) for col in color_for_points]
		brushes = [pg.mkBrush(color = self.its2col(x), ) for x in self.data]
		return brushes

	def get_pens(self):
		# color_for_points = [self.its2col(x) for x in self.data]

		# brushes = [QtGui.QBrush(QtGui.QColor(col)) for col in color_for_points]
		pens = [pg.mkPen(color = self.its2col(x), width = 4) for x in self.data]
		return pens

	def set_ranges(self):
		ranges = [35., 120., 1.2]
		zeros = [0., 40., 0.]
		self.data_plot.setYRange(zeros[self._currentData], ranges[self._currentData])

	def set_time_axis(self,
	                  plot = None):
		if plot is None:
			plot = self.data_plot
		s = self.starttime
		timeaxis = pg.AxisItem(orientation = 'bottom')
		x = numpy.linspace(0, round(len(self.data) / 60), round(len(self.data) / 60) + 1)
		ticks = ((s + x) % 24).tolist()
		ticks = [str(int(x)) + ":00" for x in ticks]
		xdict = dict(zip(x.tolist(), ticks))
		timeaxis.setTicks([xdict.items()])
		plot.centralWidget.setAxisItems({"bottom": timeaxis})

	def show_support_line(self):

		self.data_plot.addItem(self.hLine_low, ignoreBounds = True)
		self.data_plot.addItem(self.hLine_high, ignoreBounds = True)
		# self.data_plot.
		self.hLine_low.setPos(self.sup_1[self._currentData])
		self.hLine_high.setPos(self.sup_2[self._currentData])
		if self._currentData == 2:
			self.data_plot.addItem(self.hLine_idx, ignoreBounds = True)
		if self.showcrosshair:
			self.data_plot.addItem(self.hLine, ignoreBounds = True)
			self.data_plot.addItem(self.vLine, ignoreBounds = True)


if __name__ == '__main__':
	app = QApplication(sys.argv)
	myWin = LabelWindow_colored(transform = norm_transform, starttime = 23)
	myWin.show()

	sys.exit(app.exec_())
