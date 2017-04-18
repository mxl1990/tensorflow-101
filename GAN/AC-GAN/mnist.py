# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
from utils import data_to_tensor, Opt
import globals





class Mnist(object):
    r"""Downloads Mnist datasets and puts them in queues.
    """
    _data_dir = globals.mnist_dir

    def __init__(self, batch_size=128, reshape=False):

        # load sg_data set
        # one_hot参数无用，因而删减
        # read_daat_sets定义如下，
        # 只有fake_data为true的时候，one_hot参数才可用
        # read_data_sets(train_dir,
        #            fake_data=False,
        #            one_hot=False,
        #            dtype=dtypes.float32,
        #            reshape=True,
        #            validation_size=5000)
        data_set = input_data.read_data_sets(Mnist._data_dir, reshape=reshape)

        self.batch_size = batch_size

        # save each sg_data set
        _train = data_set.train
        # 验证集是直接使用训练集的前面validation_size个样本产生的
        _valid = data_set.validation
        _test = data_set.test

        # member initialize
        # Opt计算特性为加法不覆盖，乘法覆盖
        self.train, self.valid, self.test = Opt(), Opt(), Opt()

        # convert to tensor queue
        self.train.image, self.train.label = \
            data_to_tensor([_train.images, _train.labels.astype('int32')], batch_size, name='train')
        self.valid.image, self.valid.label = \
            data_to_tensor([_valid.images, _valid.labels.astype('int32')], batch_size, name='valid')
        self.test.image, self.test.label = \
            data_to_tensor([_test.images, _test.labels.astype('int32')], batch_size, name='test')

        # calc total batch count
        # 计算总共的批处理的次数
        # 每次训练batch_size个样本,总样本数除以每次的
        # 可以知道训练完整个数据集一次，需要多少次训练
        self.train.num_batch = _train.labels.shape[0] // batch_size
        self.valid.num_batch = _valid.labels.shape[0] // batch_size
        self.test.num_batch = _test.labels.shape[0] // batch_size
