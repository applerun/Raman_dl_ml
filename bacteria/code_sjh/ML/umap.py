import umap


class UMAP(umap.UMAP):
	def __init__(self,
	             *args,
	             **kwargs):
		self.__name__ = "umap"
		super().__init__(*args, **kwargs)

