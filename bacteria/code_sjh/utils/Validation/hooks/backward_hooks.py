class backward_hook():
	def __init__(self):
		self.grad_block = []
	def __call__(self, module, data_input, data_output):
		self.grad_block.append(data_output)