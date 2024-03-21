import sklearn.base
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




def gen_reduction_estimator(model = PCA,model_name = "pca", *args,
							**kwargs):
	class C:
		def __init__(self):
			self.model_name = model_name
			self.estimator = model(*args, **kwargs)

	return C

PCA_shen = gen_reduction_estimator(PCA)
LDA_shen = gen_reduction_estimator(LDA)
name2dim_reduction = {
	"lda": LDA_shen,
	"pca": PCA_shen,
}
keys = list(name2dim_reduction.values())
values = list(name2dim_reduction.keys())
dim_reduction2name = dict(zip(keys, values))


class basic_SVM():
	def __init__(self,
				 dim_reduction = None,
				 SVC_kwargs = None
				 ):
		if SVC_kwargs is None:
			SVC_kwargs = dict(gamma = 'auto', probability = True, kernel = 'rbf')
		if dim_reduction is None:
			self.model_name = "svm"
			self.dim_reduction_estimator = None
		elif type(dim_reduction) is str:
			self.model_name = dim_reduction + "_svm"
			self.dim_reduction = name2dim_reduction[dim_reduction]()
		elif dim_reduction in list(dim_reduction2name.keys()):
			self.model_name = dim_reduction2name[dim_reduction] + "_svm"
			self.dim_reduction = dim_reduction()
		elif type(dim_reduction) is type:
			self.dim_reduction = dim_reduction()
			self.model_name = self.dim_reduction.model_name + "_svm"

		self.dim_reduction_estimator = self.dim_reduction.estimator
		self.classifier = svm.SVC(**SVC_kwargs)

	def fit(self,
			x_train,
			y_train):
		if self.dim_reduction_estimator is not None:
			self.dim_reduction_estimator.fit(x_train, y_train)
			x_train_ = self.dim_reduction_estimator.transform(x_train)
		else:
			x_train_ = x_train
		self.classifier.fit(x_train_, y_train)

	def predict_proba(self,
					  x_val,
					  ):
		x_val_ = self.dim_reduction_estimator.transform(x_val) if self.dim_reduction_estimator is not None else x_val
		return self.classifier.predict_proba(x_val_)

	def score(self,
			  x_val,
			  y_val):
		x_val_ = self.dim_reduction_estimator.transform(x_val) if self.dim_reduction_estimator is not None else x_val
		return self.classifier.score(x_val_, y_val)

	def predict(self,
				x_val):
		x_val_ = self.dim_reduction_estimator.transform(x_val) if self.dim_reduction_estimator is not None else x_val
		return self.classifier.predict(x_val_)
