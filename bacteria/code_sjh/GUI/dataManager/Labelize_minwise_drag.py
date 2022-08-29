# 按分钟进行标记

from label_timewise import *
from Label_mask.label_file_handler import LabelBlock, LabelMask
from Items.Label_Mask_Item import Item_Container, Mask_Block_Item
from Widgets.labelize_wigdets_func import Label_Widget
import numpy, os
from PyQt5.Qt import QInputDialog, QLineEdit


class Label_Window_Core(LabelWindow_colored):
	def __init__(self,
	             parent = None,
	             transform = None,
	             starttime = 8):
		# self.mask_labels = ["Norm"]
		super(Label_Window_Core, self).__init__(parent, transform, starttime)
		self._currentMaskLabel = "Norm"

	# self.addlabel("Norm", )
	def closeEvent(self,
	               a0: QtGui.QCloseEvent) -> None:
		self.save_label_conf_file()  # 关闭窗口时保存mask颜色配置
		a0.accept()

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
		self.createlabel_b.clicked.connect(self.createLabelClicked)
		self.label_list.itemClicked.connect(self.labelListItemClicked)

	def createLabelClicked(self):
		value, ok = QtWidgets.QInputDialog.getText(self, "rename_label", "请输入新的label名称", QLineEdit.Normal, "New_label")
		if value in list(self.label2color.keys()) or not ok:
			return
		else:
			self.addlabel(value)

	def labelListItemClicked(self,
	                         item: QListWidgetItem):
		self._currentMaskLabel = item.text()

	def initWorkPlace(self, ):
		self.label2name = ["Norm", "Abnorm", "Abandoned"]
		self.class_num = len(self.label2name)
		# 准备存储label文件的文件夹
		if not os.path.exists(os.path.join(self.workpath, "label_mask")):
			os.makedirs(os.path.join(self.workpath, "label_mask"))
		# for l in range(self.class_num):
		# 	if not os.path.exists(os.path.join(self.workpath, self.label2name[l])):
		# 		os.makedirs(os.path.join(self.workpath, self.label2name[l]))
		# 创建缓存空间
		if not os.path.isdir(os.path.join(os.path.dirname(__file__), "cache")):
			os.makedirs(os.path.join(os.path.dirname(__file__), "cache"))
		# 读取labelfile：
		if os.path.isfile(self.label_conf_file):
			with open(self.label_conf_file, "r") as f:
				reader = csv.reader(f)
				for label, colorname in reader:
					self.label2color[label] = QtGui.QColor(colorname)
		for label in self.label2color.keys():
			color = self.label2color[label]
			self.addlabel(label, color)
		return

	def addlabel(self,
	             label,
	             color = QColor(255, 255, 255, 255)):
		item = QListWidgetItem(label)
		item.setSizeHint(QSize(180, 22))

		self.label_list.addItem(item)
		# widget = labelize_wigdets.Ui_Form()
		widget = Label_Widget(None, label, color)
		self.label_list.setItemWidget(item, widget)
		self.label2color[label] = color
		widget.reset_color_b.sigColorChanged.connect(lambda x: item.setBackground(x.color()))
		item.setBackground(QColor(color))
		widget.reset_color_b.sigColorChanged.connect(lambda x: self.update_labelcolor(widget.label, x.color()))
		widget.sigLabelChanged.connect(lambda x: self.change_labelname(list(x.keys())[0], list(x.values())[0]))
		# widget.rename_label_b.setText(label)
		# widget.reset_color_b.setColor(color)
		self.sigLabelAdded.emit(item)

		return

	def update_labelcolor(self,
	                      label,
	                      color):
		self.label2color[label] = color

	def change_labelname(self,
	                     label,
	                     new_label):
		self.label2color[new_label] = self.label2color[label]
		del self.label2color[label]
		return

	def save_label_conf_file(self):
		with open(self.label_conf_file, "w", newline = "") as f:
			writer = csv.writer(f)
			for label in self.label2color.keys():
				writer.writerow([label, self.label2color[label].name()])


class Label_Window_Mask(Label_Window_Core):
	def __init__(self,
	             parent = None,
	             transform = None,
	             starttime = 8):
		super(Label_Window_Mask, self).__init__(parent, transform, starttime)
		p2 = self.data_plot.centralWidget
		vb = p2.vb
		proxy = pg.SignalProxy(p2.scene().sigMouseMoved, rateLimit = 60, slot = self.mouseMoved)
		self.floatRegionItemRegion = None
		self.readMaskFile()
		self.sigDataChanging.connect(self.save_mask)
		self.sigDataChanged.connect(self.readMaskFile)
		self.initMaskButton()


	def initMaskButton(self):
		self.labelModeChanege_btn.clicked.connect(self.labelModeChange_btn_clicked)
		self.addmask_b.clicked.connect(self.addLabelMask_btn_clicked)

	def readMaskFile(self):
		self.initMaskItems()
		self.item_container.floatRegionItemRegion = (self.item_container[0].start, self.item_container[-1].end)
		if self._currentFile is None:
			return
		maskfile = os.path.basename(self._currentFile)[:-4]
		maskfile = os.path.join(self.workpath, "label_mask", maskfile)
		if os.path.exists(maskfile):
			self.item_container.load(maskfile)
			self.item_container.regions_update()
		return

	def save_mask(self,
	              ):  # 保存mask
		# data = self.data
		# filename = self._currentFile[:-4]
		# with open(os.path.join(self.))
		if self._currentFile is None:
			return
		maskfile = self._currentFile[:-4]
		maskfile = os.path.join(self.workpath, "label_mask", maskfile)
		self.item_container.save(maskfile)
		if self.item_container.mode == 0:
			self.floatRegionItemRegion = self.item_container.floatRegionItem.getRegion()
		self.item_container.destroy_self()
		return
	def mouseMoved(self,
	               evt):
		pos = evt[0]  ## using signal proxy turns original arguments into a tuple

		if self.data_plot.sceneBoundingRect().contains(pos):
			vb = self.data_plot.centralWidget.vb
			mousePoint = vb.mapSceneToView(pos)
			x = float(mousePoint.x())
			# x_idx = int(float(mousePoint.x()) * 60)
			if x >= 0 and x < self.item_container[-1].end:
				time = float(mousePoint.x()) + self.starttime
				time = time % 24
				h = int(time)
				m = (time - h) * 60
				pointlabel = self.item_container.label(x)
				time = "{},{}:{:0>2}".format(pointlabel, h, int(m), )
				self.poslabel.setText(str(time))
				self.vLine.setPos(mousePoint.x())
				self.hLine.setPos(mousePoint.y())

	def initMaskItems(self):
		self.item_container = Item_Container(self.data_plot,
		                                     (0, self.data_plot_item.xData[-1]),
		                                     float,
		                                     clipItem = self.data_plot_item,
		                                     floatRegionItemRegion_default = self.floatRegionItemRegion
		                                     )
		self.item_container.updatBrushesByColor(self.label2color)

	def labelModeChange_btn_clicked(self):
		if self.item_container.mode == 0:
			self.floatRegionItemRegion = self.item_container.floatRegionItem.getRegion()
		self.item_container.changeMode()

	def addLabelMask_btn_clicked(self):
		label = self._currentMaskLabel
		block = LabelBlock(self.item_container.floatRegionItem.getRegion(), label = label)
		self.item_container.addMask(block, brush = pg.mkBrush(self.label2color[label]))


if __name__ == '__main__':
	app = QApplication(sys.argv)
	myWin = Label_Window_Mask(transform = norm_transform, starttime = 23)
	myWin.show()

	sys.exit(app.exec_())
