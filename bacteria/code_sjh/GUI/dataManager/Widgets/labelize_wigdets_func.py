try:
	from bacteria.code_sjh.GUI.dataManager.Widgets import labelize_wigdets
except:
	from bacteria.code_sjh.GUI.dataManager.Windows.labelize_minwise import *
	import bacteria.code_sjh.GUI.dataManager.Widgets.labelize_wigdets
from PyQt5 import QtWidgets


class Label_Widget(QtWidgets.QWidget, labelize_wigdets.Ui_Form):

	def __init__(self,
	             parent = None,
	             label = "Norm",
	             color = "#ffffff"):
		super(Label_Widget, self).__init__(parent = parent)
		self.setupUi(self)
		self.reset_color_b.sigColorChanged.connect(lambda x: self.reset_color(x.color().name()))

		self.rename_label_b.setText(label)
		self.reset_color_b.setColor(color)
		self.color = color
		self.label = self.rename_label_b.text()
		self.rename_label_b.clicked.connect(self.rename_label_btn_clicked)


	def rename_label_btn_clicked(self):
		txt = QtWidgets.QInputDialog.getText(self, "rename_label", "请输入新的label名称", "New_label")
		self.rename_label(txt)

	def reset_color(self,
	                color):
		self.color = color

	def rename_label(self,
	                 label):
		self.rename_label_b.setText(label)
		self.label = label

