import os
import shutil


# 获得根路径
def getRootPath(projectname):
	# 获取文件目录
	curPath = os.path.abspath(os.path.dirname(__file__))
	# 获取项目根路径，内容为当前项目的名字
	rootPath = curPath[:curPath.find(projectname) + len(projectname)]
	return rootPath


def copy2merge(src, tgt):
	if not os.path.isdir(tgt):
		shutil.copytree(src, tgt)
		return

	for f in os.listdir(src):
		src_path = os.path.join(src, f)
		tgt_path = os.path.join(tgt, f)
		if os.path.isfile(src_path):
			if os.path.isfile(tgt_path):
				continue
			else:
				shutil.copy(src_path, tgt_path)
		elif os.path.isdir(src_path):
			if os.path.isdir(tgt_path):
				copy2merge(src_path, tgt_path)
			else:
				shutil.copytree(src_path, tgt_path)
