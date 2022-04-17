from sklearn import svm
import os, time, csv
import numpy as np
from bacteria.code_sjh.utils.RamanData import Raman, projectroot, getRamanFromFile,Raman_dirwise

if __name__ == '__main__':
	raman = Raman_dirwise
	k_split = 10
	readdatafunc0 = getRamanFromFile(wavelengthstart = 390, wavelengthend = 1810,
	                                 dataname2idx = {"Wavelength": 0, "Column": 2, "Intensity": 1},
	                                 delimeter = "\t", )
	from scipy import interpolate


	def readdatafunc(
			filepath
	):
		R, X = readdatafunc0(filepath)
		R = np.squeeze(R)
		f = interpolate.interp1d(X, R, kind = "cubic")
		newX = np.linspace(400, 1800, 512)
		newR = f(newX)
		newR = np.expand_dims(newR, axis = 0)

		return newR, newX
	db_cfg = dict(dataroot = os.path.join(projectroot, "data", "liver_cell"),
	              LoadCsvFile = readdatafunc,
	              backEnd = ".asc", t_v_t = [1., 0., 0.], newfile = False, k_split = k_split,
	              )
	              # transform = None, )

	recordfile = "SVM_Record" + time.asctime().replace(":", "-").replace(" ", "_") + ".csv"
	f = open(recordfile, "w", newline = "")
	writer = csv.writer(f)
	f.write(db_cfg.__str__() + "\n")
	writer.writerow(["n", "k", "test_acc"])
	testaccs = []
	c = 0
	for n in range(1):
		for k in range(k_split):
			sfpath = "Raman_" + str(n) + ".csv"
			train_db = raman(**db_cfg, mode = "train", k = k, sfpath = sfpath)
			val_db = raman(**db_cfg, mode = "val", k = k, sfpath = sfpath)

			if not n and not k:
				train_db.show_data(win = "train")
				val_db.show_data(win = "test")
			l1 = train_db.RamanFiles
			l2 = val_db.RamanFiles
			l = list(set(l1)&set(l2))
			print(len(l)/len(val_db))
			train_data, train_label = [np.squeeze(x.numpy()) for x in train_db.Ramans], [x.item() for x in
			                                                                             train_db.labels]
			test_data, test_label = [np.squeeze(x.numpy()) for x in val_db.Ramans], [x.item() for x in val_db.labels]

			classifier = svm.SVC(C = 2, kernel = 'rbf', gamma = 10, decision_function_shape = 'ovr')  # ovr:一对多策略
			classifier.fit(train_data, train_label)

			test_acc = classifier.score(test_data, test_label)
			testaccs.append(test_acc)
			writer.writerow([n, k, test_acc])
			c+=1
			print(c,"/",1*k_split)
	ta = np.mean(np.array(testaccs)).__str__() + "+-" + np.std(np.array(testaccs)).__str__()
	writer.writerow(["mean", "std", ta])
