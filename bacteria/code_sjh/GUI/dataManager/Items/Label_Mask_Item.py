import warnings

import PyQt5.QtGui
from PyQt5 import QtCore, QtGui, QtWidgets

import pyqtgraph as pg

try:
	from ..Label_mask.label_file_handler import LabelMask, LabelBlock
except:
	from bacteria.code_sjh.GUI.dataManager.Label_mask.label_file_handler import LabelMask, LabelBlock


class Mask_Block_Item(pg.LinearRegionItem):
	sigRegionChangeFinished = QtCore.pyqtSignal(object)
	sigRegionChanged = QtCore.pyqtSignal(object)
	sigRegionChangeFinished_output = QtCore.pyqtSignal(float, float)
	sigCheckContainer = QtCore.pyqtSignal(object)

	def __init__(self,
	             block: LabelBlock = None,
	             **kwargs):
		super(Mask_Block_Item, self).__init__(**kwargs)
		if block is None:
			self.start, self.end = self.getRegion()
			self.label = None
			self.register_block(None)
		else:
			self.start, self.end, self.label = block[0], block[1], block[2]
			self.setRegion([self.start, self.end])
			self.register_block(block)
		self.sigRegionChanged.connect(self.ref_par)
		self.sigRegionChangeFinished.connect(self.regionChangeFinished)
	def regionChangeFinished(self):
		self.sigRegionChangeFinished_output.emit(self.start, self.end)
		self.sigCheckContainer.emit(self)

	def ref_par(self, ):
		self.start, self.end = self.getRegion()

	def register_block(self,
	                   block: LabelBlock):
		self.registered_block = block


class Item_Container(LabelMask):
	def __init__(self,
	             PlotWidget,
	             data_range: tuple,
	             precision = float,
	             normlabel = "Norm",
	             clipItem = None):
		super(Item_Container, self).__init__(data_range, precision, normlabel = normlabel)
		self.regionItems = []
		self.PlotWidget = PlotWidget
		self.mode = 0  # 0 - 添加模式；1 - 调整模式
		self.clipItem = clipItem  # 设置本Container中所有Region的clipItem
		self.create_floatRegionItem()  # 如果为添加模式，则创建一个用于添加label的region
		self.regions_update()
		self.label2brush = {"Norm": None}

	def changeBrush(self,
	                input):
		if type(input) == dict:  # 直接更新label2brush
			self.label2brush.update(input)
		self.regions_update()
		return

	def create_floatRegionItem(self):
		self.floatRegionItem = Mask_Block_Item(movable = True, )
		# region = (self.blocks[0][0], self.blocks[-1][1])
		self.PlotWidget.addItem(self.floatRegionItem,
		                        ignoreBounds = True, )
		self.floatRegionItem.setClipItem(self.clipItem)

	def del_floatRegionItem(self):
		self.PlotWidget.removeItem(self.floatRegionItem)
		del self.floatRegionItem

	def delitem_key(self,
	                key):
		assert len(self.blocks) == len(self.regionItems)
		self.PlotWidget.removeItem(self.regionItems[key])
		del self.blocks[key]
		del self.regionItems[key]

	def delitem_label(self,
	                  label):
		assert len(self.blocks) == len(self.regionItems)
		for i in range(len(self.blocks)):
			assert self.blocks[i].label == self.regionItems[i].label
			if self.blocks[i].label == label:
				self.PlotWidget.removeItem(self.regionItems[i])
				del self.blocks[i]
				del self.regionItems[i]

	def check_self(self):
		assert len(self.blocks) == len(self.regionItems)
		for i in range(len(self)):
			block = self.blocks[i]
			region = self.regionItems[i]
			assert block.start == region.start
			assert block.end == region.end
			assert block.label == region.label

	def checkMovedRegionItem(self,
	                         item: Mask_Block_Item):

		item.registered_block.label = self.normlabel
		self.update(item.registered_block) # 抹去原本的block
		self.update(LabelBlock((item.start,item.end),item.label)) # 更新新的block
		self.sort()
		self.regions_update()
		return

	def blocks_update(self):
		self.blocks = [LabelBlock((self.blocks[0][0], self.blocks[-1][1]), label = self.normlabel)]
		for regionItem in self.regionItems:
			self.update(LabelBlock((regionItem.start, regionItem.end), regionItem.label))
		return

	def regions_update(self):

		for item in self.regionItems:
			self.PlotWidget.removeItem(item)
			del item
		self.regionItems = []
		for b in self.blocks:
			if b.label == self.normlabel:
				continue
			self.createRegion(b, self.label2brush[b.label]) # .sigRegionChangeFinished_output.connect(b.changeRegion)

		return

	def createRegion(self,
	                 block,
	                 brush, ):
		new_item = Mask_Block_Item(block = block, brush = brush, swapMode = "push",
		                           movable = False if self.mode == 0 else True)
		self.regionItems.append(new_item)
		self.PlotWidget.addItem(new_item)
		new_item.setClipItem(self.clipItem)
		new_item.sigCheckContainer.connect(self.checkMovedRegionItem)
		return new_item

	def loadmask(self,
	             file):
		self.load(file, raisewarnings = False)
		for b in self.blocks:
			if not b.label in list(self.label2brush.keys()):
				warnings.warn("brush of label {} lost,use default color".format(b.label))
				self.label2brush[b.label] = pg.mkBrush(0, 0, 255, 55)
		self.regions_update()

	def changeMode(self,
	               mode = None):
		if mode is None:
			mode = 1 - self.mode
		elif mode not in [0, 1]:
			raise "wrong mode:{}".format(mode)

		if mode == self.mode:
			# TODO:更改所有bounder的操作方式
			return
		elif mode == 0:
			self.setMovable(False)

		elif mode == 1:
			self.del_floatRegionItem()
			self.setMovable(True)

	def setMovable(self,
	               m = True):
		self.mode = 1 if m else 0
		for items in self.regionItems:
			items.setMovable(m)

	def addMask(self,
	            block,
	            brush: PyQt5.QtGui.QBrush = None
	            # brush = pg.mkBrush("#55abcd")
	            ):  # 添加mask区域

		if not brush is None:
			if brush.color().alpha() > 70:
				brush.color().setAlpha(70)
			self.label2brush[block.label] = brush

		else:
			if not block.label in list(self.label2brush.keys()):
				self.label2brush[block.label] = pg.mkBrush(255, 0, 0, 70)

		# if type(block) == LabelBlock:
		# 	item = Mask_Block_Item(block, values = (block[0], block[1]), swapMode = swapmode, brush = brush,
		# 	                       movable = False if self.mode == 1 else True)
		#
		# self.PlotWidget.addItem(item, ignoreBounds = True)
		# item.setClipItem(self.clipItem)
		# item.setBrush(brush)
		# item.swapMode = swapmode
		# self.regionItems.append(item)
		self.update(block)
		self.regions_update()

	def setClipItem(self,
	                item):  # set an item to which regions is bounded
		if item is self.clipItem:
			return
		self.clipItem = item
		for item in self.regionItems:
			item.setClipItem(self.clipItem)
		if self.mode == 0:
			self.floatRegionItem.setClipItem(self.clipItem)
		return


if __name__ == '__main__':
	import numpy as np

	data1 = 10000 + 15000 * pg.gaussianFilter(np.random.random(size = 10000), 10) + 3000 * np.random.random(
		size = 10000)

	app = pg.mkQApp("Crosshair Example")
	win = pg.GraphicsLayoutWidget(show = True)
	win.setWindowTitle('pyqtgraph example: crosshair')
	label = pg.LabelItem(justify = 'right')
	win.addItem(label)

	p2 = win.addPlot(row = 1, col = 0)
	vb = p2.vb

	p2d = p2.plot(data1, pen = "w")
	c = Item_Container(p2, (0, 120000), clipItem = p2d)

	c.changeMode()


	def mouseMoved(evt):
		pos = evt[0]  ## using signal proxy turns original arguments into a tuple
		if p2.sceneBoundingRect().contains(pos):
			mousePoint = vb.mapSceneToView(pos)
			index = int(mousePoint.x())
			if index > 0 and index < len(data1):
				label.setText(c.label(index))


	proxy = pg.SignalProxy(p2.scene().sigMouseMoved, rateLimit = 60, slot = mouseMoved)
	c.addMask(LabelBlock((2000, 6000), label = "A", ), brush = pg.mkBrush(155, 120, 50, 50))
	c.addMask(LabelBlock((500, 1500), label = "B"))
	c.addMask(LabelBlock((10000,10000),label = "ts"))
	pg.exec()
