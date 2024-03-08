from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn import svm


def preprocess_PCA(x_train,
                   y_train,
                   x_test,
                   y_test):
	pca = PCA(n_components = 2)
	pca.fit(x_train)
	x_train = pca.transform(x_train)
	x_test = pca.transform(x_test)
	return x_train, y_train, x_test, y_test


def preprocess_LDA(x_train,
                   y_train,
                   x_test,
                   y_test):
	lda = LDA(n_components = 1)
	lda.fit(x_train, y_train)
	x_train = lda.transform(x_train)
	x_test = lda.transform(x_test)
	return x_train, y_train, x_test, y_test


name2dim_reduction = {
	"lda": LDA,
	"PCA": PCA,
}
keys = list(name2dim_reduction.values())
values = list(name2dim_reduction.keys())
dim_reduction2name = dict(zip(keys, values))


class PCA(PCA):
	def __init__(self,
	             *args,
	             **kwargs):
		self.__name__ = "pca"
		super().__init__(*args, **kwargs)


class LDA(LDA):
	def __init__(self,
	             *args,
	             **kwargs):
		self.__name__ = "lda"
		super().__init__(*args, **kwargs)


class basic_SVM():
	def __init__(self,
	             dim_reduction = None,
	             reducted_n_components = None,
	             ):
		if dim_reduction is None:
			self.__name__ = "svm"
			self.dim_reduction = dim_reduction
		elif type(dim_reduction) is str:
			self.__name__ = dim_reduction + "_svm"
			dim_reduction = name2dim_reduction[dim_reduction]
		elif dim_reduction in list(dim_reduction2name.keys()):
			self.__name__ = dim_reduction2name[dim_reduction] + "_svm"
		else:
			self.__name__ = dim_reduction.__name__ + "_svm"

		if dim_reduction is not None:
			if type(dim_reduction) is type:
				self.dim_reduction = dim_reduction(n_components = reducted_n_components)
			else:
				self.dim_reduction = dim_reduction

		self.classifier = svm.SVC(C = 2, kernel = 'rbf', gamma = 10, decision_function_shape = 'ovr',
		                          probability = True)

	def fit(self,
	        x_train,
	        y_train):
		if self.dim_reduction is not None:
			self.dim_reduction.fit(x_train, y_train)
			x_train_ = self.dim_reduction.transform(x_train)
		else:
			x_train_ = x_train
		self.classifier.fit(x_train_, y_train)

	def predict_proba(self,
	                  x_val,
	                  ):
		x_val_ = self.dim_reduction.transform(x_val) if self.dim_reduction is not None else x_val
		return self.classifier.predict_proba(x_val_)

	def score(self,
	          x_val,
	          y_val):
		x_val_ = self.dim_reduction.transform(x_val) if self.dim_reduction is not None else x_val
		return self.classifier.score(x_val_, y_val)

	def predict(self,
	            x_val):
		x_val_ = self.dim_reduction.transform(x_val) if self.dim_reduction is not None else x_val
		return self.classifier.predict(x_val_)
