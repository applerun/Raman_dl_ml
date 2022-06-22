import numpy
import numpy as np


def all_eval_err_handle(data: np.ndarray,
                        err_cri = None):
	if err_cri is None:
		err_cri = lambda key: True if key == 0 else False
	err = numpy.vectorize(err_cri)(data)
	err_ = err == False
	if np.sum(err_) == 0:
		return data
	eval = np.sum(err_ * data) / np.sum(err_)
	res = data+err*eval
	return res

if __name__ == '__main__':
    inp = np.array([0,0,0,0,0])
    print(all_eval_err_handle(inp))