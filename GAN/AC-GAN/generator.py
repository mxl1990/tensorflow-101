import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope


# 大致的可以认为这个generator
# 有两层隐藏层，分别为1024个节点和7*7*128个节点
# 在两层隐藏层后，是两次卷积层
def generator(tensor):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('generator')]) > 0
    # print(tensor.get_shape())
    # 用于生成上下文管理器，来管理变量名
    # 对应optimizer里面
    # with里面所有创建的变量，其name属性都由generator开头
    with variable_scope.variable_scope('generator', reuse = reuse):
        # fully_connected函数定义在tensorflow.contrib.layers.python.layers中
        # 创建全连接神经网络，tensor为输入
        # 1024为输出的个数
        tensor = slim.fully_connected(tensor, 1024)

        # Batch Normalization
        tensor = slim.batch_norm(tensor, activation_fn=tf.nn.relu)

        tensor = slim.fully_connected(tensor, 7*7*128)

        tensor = slim.batch_norm(tensor, activation_fn=tf.nn.relu)

        # 变换tensor的shape
        # 这里变成32*7*7*128的tensor
        tensor = tf.reshape(tensor, [-1, 7, 7, 128])

        # [batch, height, width, in_channels]
        # 计算卷积
        tensor = slim.conv2d_transpose(tensor, 64, kernel_size=[4,4], stride=2, activation_fn = None)
        tensor = slim.batch_norm(tensor, activation_fn = tf.nn.relu)
        tensor = slim.conv2d_transpose(tensor, 1, kernel_size=[4, 4], stride=2, activation_fn=tf.nn.sigmoid)
    return tensor