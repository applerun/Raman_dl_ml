from label_timewise import *
from Label_mask.label_file_handler import LabelBlock, LabelMask
from Items.Label_Mask_Item import Item_Container, Mask_Block_Item
from Widgets.labelize_wigdets_func import Label_Widget
import numpy, os


class Label_Window_Drag_Mask(LabelWindow_colored):
	def __init__(self,
	             parent = None,
	             transform = None,
	             starttime = 8):
		super(Label_Window_Drag_Mask, self).__init__(parent, transform, starttime)

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
		# 读取labelfile：
		if os.path.isfile(self.label_conf_file):
			with open(self.label_conf_file, "r") as f:
				reader = csv.reader(f)
				for label, color in reader:
					self.label2color[label] = color
		for label in self.label2color.keys():
			color = self.label2color[label]
			self.addlabel(label, color)
		return



	def createLabelContainer(self, ):

		self.Container = Item_Container(self.data_plot, (self.starttime, len(self.data) / 60),
		                                clipItem = self.data_plot.plotItem)

	def addlabel(self,
	             label,
	             color):
		item = QListWidgetItem(label)
		item.setSizeHint(QSize(165, 22))

		self.label_list.addItem(item)
		# widget = labelize_wigdets.Ui_Form()
		widget = Label_Widget(None, label, color)
		self.label_list.setItemWidget(item, widget)
		self.label2color[label] = color

		widget.reset_color_b.sigColorChanged.connect(lambda x: item.setBackground(x.color()))

		item.setBackground(QColor(color))
		# widget.rename_label_b.setText(label)
		# widget.reset_color_b.setColor(color)
		self.sigLabelAdded.emit(item)
		return
