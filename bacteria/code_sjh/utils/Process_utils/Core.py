import numpy
import copy
import torch

ctype2cfunc = {"copy": copy.copy, "deepcopy": copy.deepcopy}


class ProcessorRoot:
    def __init__(self):
        return

    def __call__(self, y, x = None):
        return y if x is None else (y, x)

    def __str__(self):
        return ""

    def __add__(self, other):
        return


class ProcessorRootSeries(ProcessorRoot):
    def __init__(self, sequence):
        """

        @param sequence: list of Processors

        """

        super(ProcessorRootSeries, self).__init__()

        self.sequence = list(filter(lambda x: len(str(x)) > 0, sequence))
        self.str = " --> ".join([str(x) for x in self.sequence])
    def __call__(self, y, x = None):
        for funcs in self.sequence:
            y = funcs(y, x)
            if x is not None:
                y, x = y
        return y if x is None else (y, x)

    def __str__(self):
        return self.str

    def __add__(self, other: ProcessorRoot):
        new_processor = ProcessorRootSeries([self, other])
        return new_processor


class ProcessorFunction(ProcessorRoot):
    """
    用于基本的Function,默认为不操作
    """

    def __init__(self, name = ""):
        super(ProcessorFunction, self).__init__()
        self.name = name

    def __call__(self, y, x = None):
        return y if x is None else (y, x)

    def __add__(self, other):
        new_processor = ProcessorRootSeries([self, other])
        return new_processor

    def __str__(self):
        return self.name


class DeepCopy(ProcessorFunction):
    """
    数据深拷贝（不在原来的数据上进行操作
    """

    def __init__(self, ):
        super(DeepCopy, self).__init__("")

    def __call__(self, y, x = None):
        return copy.deepcopy(y) if x is None else (copy.deepcopy(y), x)


class pytorch_process(ProcessorRoot):
    def __init__(self, process_func: ProcessorRoot):
        super(pytorch_process, self).__init__()
        self.func = process_func

    def __call__(self, y, x = None):
        y = y.numpy()
        res = self.func(y, x)
        if x is None:
            return torch.Tensor(res)
        else:
            return torch.Tensor(res[0]), res[1]

    def __add__(self, other):
        new_processor = ProcessorRootSeries([self.func, other.func if type(other) == pytorch_process else other])
        return pytorch_process(new_processor)

    def __str__(self):
        return str(self.func)


class batch_process(ProcessorRoot):
    def __init__(self, process_func: ProcessorRoot, copytype = "deepcopy", verbose = False):
        super(batch_process, self).__init__()
        self.copytype = copytype
        self.process_func = process_func
        self.verbose = verbose

    def __call__(self, y, x = None):
        res_y = None
        res_x = None
        y = ctype2cfunc[self.copytype](y)
        y = numpy.squeeze(y)
        if len(y.shape) == 1:
            return self.process_func(y, x)

        elif len(y.shape) == 2:
            batch_size = y.shape[0]
        else:
            raise AssertionError
        for line_i in range(batch_size):
            if self.verbose:
                if line_i > 0:
                    print("\r", end = "")
                print(line_i + 1, "/", batch_size, end = "" if not line_i == batch_size - 1 else "\n")

            if x is None:
                temp = self.process_func(y[line_i, :])
                y[line_i, :] = temp
                res_x = x
                continue
            else:
                y_, x_ = self.process_func(y[line_i, :], x)
                if res_x is None:
                    res_x = x_
                else:
                    assert len(res_x) == len(x_)
            if res_y is None:
                res_y = numpy.expand_dims(y_, 0)
            else:
                res_y = numpy.vstack((res_y, numpy.expand_dims(y_, 0)))

        return y if x is None else (res_y, res_x)

    def __add__(self, other):
        new_processor = ProcessorRootSeries(
            [self.process_func, other if type(other) in (ProcessorFunction, ProcessorRootSeries) else other.func])
        return batch_process(new_processor)

    def __str__(self):
        return str(self.process_func)


