import numpy

from bacteria.code_sjh.utils.RamanData import RamanDatasetCore
from bacteria.code_sjh.models import BasicModule
from bacteria.code_sjh.utils.Validation.validation import grad_cam
import os
import torch
import shutil


def copy_filewise_classify(db: RamanDatasetCore,
                           model: BasicModule,
                           dst: str,
                           device: torch.device = None):
	model.eval()
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	root = db.root
	if not os.path.isdir(dst):
		os.makedirs(dst)
	model = model.to(device)
	label2name = db.label2name()
	for i in range(len(db)):
		raman, label = db[i]
		raman = torch.unsqueeze(raman, dim = 0).to(device)

		file = db.RamanFiles[i]
		src = os.path.join(root, file)
		filename = file.split(os.sep)[-1]
		with torch.no_grad():
			logits = model(raman)  # [1,n_c]

		pred = torch.squeeze(logits.argmax(dim = 1)).item()
		true_name = label2name[label.item()]
		pred_name = label2name[pred]
		dst_dir = os.path.join(dst, true_name + "2" + pred_name)
		if not os.path.isdir(dst_dir):
			os.makedirs(dst_dir)
		dst_p = os.path.join(dst_dir, filename)
		shutil.copy(src, dst_p)


def cam_output_filewise(db: RamanDatasetCore,
                        model: BasicModule,
                        dst: str,
                        device: torch.device = None):
	model.train()
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# root = db.root
	if not os.path.isdir(dst):
		os.makedirs(dst)
	model = model.to(device)
	# label2name = db.label2name()
	if not os.path.isdir(dst):
		os.makedirs(dst)
	for i in range(len(db)):
		raman, label = db[i]
		raman = torch.unsqueeze(raman, dim = 0).to(device)
		file = db.RamanFiles[i]
		filename = file.split(os.sep)[-1][:-4] + ".cam.csv"
		cam = grad_cam(model,raman,label = None)
		dst_p = os.path.join(dst, filename)
		numpy.savetxt(dst_p,cam,delimiter = ",")

