import copy
import csv
import warnings

import numpy as np
import os


class LabelBlock():
	def __init__(self,
	             region: tuple,
	             label = None,
	             precision = float):
		self.precision = precision
		if label is None:
			label = "Norm"
		self.start, self.end = region
		self.start = self.start if type(self.start) == self.precision else self.precision(self.start)
		self.end = self.end if type(self.end) == self.precision else self.precision(self.end)
		if not type(label) == str:
			self.label = str(label)
		else:
			self.label = label

	def __contains__(self,
	                 item):
		if type(item) == LabelBlock:
			return self.start < item.start < item.end < self.end
		else:
			return self.start < item < self.end

	def __add__(self,
	            other):
		if self.label != other.label:
			warnings.warn("label of {} if not the same with {}".format(other, self))
		if self.start > other.end or self.end < other.start:
			raise ValueError
		self.start = min(self.start, other.start)
		self.end = max(self.end, other.end)
		return self

	def __eq__(self,
	           other):
		if self.label != other.label:
			warnings.warn("label of {} if not the same with {}".format(other, self))
		return self.start == other.start and self.end == other.end

	def __lt__(self,
	           other):
		return self.start < self.end <= other.start < other.label

	def __int__(self):
		if self.precision == int:
			return self
		self.precision = int
		self.start = int(self.start)
		self.end = int(self.end)
		return self

	def __float__(self):
		if self.precision == float:
			return self
		self.precision = float
		self.start = float(self.start)
		self.end = float(self.end)
		return self

	def cut(self,
	        mid = None,
	        cut_prev = False):
		if mid is None:
			mid = (self.start + self.end) / 2
		if not type(mid) == self.precision:
			mid = self.precision(mid)
		assert mid in self, "mid = {} out of range - {}:{}".format(mid, self.start, self.end)
		if cut_prev:
			newblock = LabelBlock((self.start, mid), label = self.label)
			self.start = mid
		else:
			newblock = LabelBlock((mid, self.end), label = self.label)
			self.end = mid
		return self, newblock

	def __getitem__(self,
	                item):
		return [self.start, self.end, self.label][item]

	def __str__(self):
		return "{},{},{}".format(self.start, self.end, self.label)


class LabelMask():
	def __init__(self,
	             data_range: tuple,
	             precision = float,
	             normlabel = "Norm"):
		self.blocks = [
			LabelBlock(data_range, precision = precision, label = normlabel)
		]
		self.precision = precision
		self.historys = []
		self.normlabel = normlabel

	def __len__(self):
		return len(self.blocks)

	def __getitem__(self,
	                item):
		return self.blocks[item]

	def __delitem__(self,
	                key):
		del self.blocks[key]

	def __add__(self,
	            other):
		res = copy.deepcopy(self)
		for i in range(len(other)):
			new_block = other[i]
			res.update(new_block)
		return res

	def __int__(self):
		self.precision = int
		self.blocks = [int(x) for x in self.blocks]
		return self

	def __float__(self):
		self.precision = float
		self.blocks = [float(x) for x in self.blocks]
		return self

	def __str__(self):
		return "[(" + ")\n(".join([str(x) for x in self.blocks]) + ")]"

	def sort(self):
		self.blocks = sorted(self.blocks)
		end_ = self.blocks[0][0]
		for i in range(len(self)):
			start, end = self[i][0], self[i][1]
			assert start == end_, "start-{},end-{}".format(start, end_)
			end_ = end
		i = 0
		while i < len(self) - 1:
			if self[i].label == self[i + 1].label:
				self.blocks[i] += self.blocks[i + 1]
				del self[i + 1]
			i += 1

	def save(self,
	         file):
		dir = os.path.dirname(file)
		if not os.path.isdir(dir):
			os.makedirs(dir)
		with open(file, "w", ) as f:
			f.writelines([str(self[i]) for i in range(self.__len__())])

	def load(self,
	         file,
	         raisewarnings = False
	         # 是否异常数据报警
	         ):
		with open(file, "r", ) as f:
			end_ = self.blocks[0][0]
			end__ = self.blocks[-1][1]
			label_ = self.blocks[0][-1]
			self.blocks = []

			for lines in f.readlines():
				start, end, label = lines.split(",")
				start = self.precision(start)
				end = self.precision(end)

				if start < end_:
					if label != label_ and raisewarnings:
						warnings.warn("label mask conflict: from {} to {},chose {} abandoned {}".format(
							start, end_, label, label_
						))
					start = end_
				if start > end_ and raisewarnings:
					warnings.warn("label mask lost:from {} to {}".format(end_, start))
				self.blocks.append(
					LabelBlock((start, end), label)
				)
				end_ = end
				label_ = label
			if end_ < end__ and raisewarnings:
				warnings.warn("label mask lost:from {} to {},check ranges".format(end_, end__))
			if end_ > end__ and raisewarnings:
				warnings.warn("data out of range:{} higher than {},check ranges".format(end_, end__))

	def update(self,
	           new_block: LabelBlock):
		self.historys.append(copy.deepcopy(self.blocks))
		new_s, new_e = new_block.start, new_block.end
		new_l = new_block.label
		flag0 = False
		i = 0
		while len(self) > i:
			b = self[i]
			if not flag0:  # 未找到新的block的位置
				if new_s == b.start:
					flag0 = True
					continue
				if new_s in b:
					b, b_ = b.cut(new_s)
					self.blocks.insert(i + 1, b_)
					flag0 = True
			elif flag0:  # 此时new_s 为某一block的起点
				if new_e in b:
					b, b_ = b.cut(new_e)
					b.label = new_l
					self.blocks.insert(i + 1, b_)
					break
				elif new_e == b.start:
					continue
				elif new_e == b.end:
					b.label = new_l
					break
				elif new_e > b.end:
					b.label = new_l
				else:
					raise ValueError

			i += 1
		self.sort()

	def label(self,
	          x):
		assert self[0][0] < x < self[-1][1], "x({}) out of range:{} to {}".format(x, self[0][0], self[-1][1])
		for b in self.blocks:
			if x in b:
				return b.label
		return None

	def get_label(self,
	              x):  # 返回坐标对应的标签，超出范围记为None
		for i in range(len(self)):
			if x in self[i]:
				return self[i].label
		return None


if __name__ == '__main__':
	b = LabelBlock((1, 2), "b")
	print(b)
	m = LabelMask((0, 15))
	print(m)
	m.update(b)
	print(m)
	blocklist = [LabelBlock((1, 1.5)), LabelBlock((1.75, 10), "d")]
	for bs in blocklist:
		m.update(bs)
		print(m)
