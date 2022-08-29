try:
	from bacteria.code_sjh.GUI.dataManager.Widgets import labelize_wigdets
except:
	from bacteria.code_sjh.GUI.dataManager.Windows.labelize_minwise import *
	import bacteria.code_sjh.GUI.dataManager.Widgets.labelize_wigdets
from PyQt5 import QtWidgets, QtCore,QtGui
from PyQt5.QtGui import QColor
from PyQt5.Qt import QInputDialog, QLineEdit

class Label_Widget(QtWidgets.QWidget, labelize_wigdets.Ui_Form):
	sigColorChanged_Brush = QtCore.pyqtSignal(dict)  # 改变颜色时发出{Label ：颜色对应的笔刷}
	sigLabelChanged = QtCore.pyqtSignal(dict)  # 改变label时发出：{原label，新label}

	def __init__(self,
	             parent = None,
	             label = "Norm",
	             color = QColor(255,255,255,255)):
		super(Label_Widget, self).__init__(parent = parent)
		self.setupUi(self)
		self.reset_color_b.sigColorChanged.connect(lambda x: self.reset_color(x.color()))
		self.rename_label_b.setText(label)
		self.label = label
		self.color = color
		self.reset_color_b.setColor(color)

		self.label = self.rename_label_b.text()
		self.rename_label_b.clicked.connect(self.rename_label_btn_clicked)

	def rename_label_btn_clicked(self):
		txt,ok = QtWidgets.QInputDialog.getText(self, "rename_label", "请输入新的label名称",QLineEdit.Normal, "New_label")
		if ok:
			self.rename_label(txt)

	def reset_color(self,
	                color):
		self.color = color
		self.sigColorChanged_Brush.emit({self.label: QtGui.QBrush(self.color)})

	def rename_label(self,
	                 label):
		self.sigLabelChanged.emit({self.label: label})
		self.rename_label_b.setText(label)
		self.label = label
