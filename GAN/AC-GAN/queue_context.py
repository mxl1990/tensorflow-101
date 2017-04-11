# -*- coding: utf-8 -*-
import tensorflow as tf

def queue_context(sess=None):
    r"""Context helper for queue routines.

    Args:
      sess: A session to open queues. If not specified, a new session is created.

    Returns:
      None
    """

    # default session
    # 返回当前线程的默认会话
    sess = tf.get_default_session() if sess is None else sess

    # thread coordinator
    # 创建多线程的协调器
    coord = tf.train.Coordinator()
    # 在sess指定的会话，用coord指定的协调器启动所有queue_runner
    # 返回所有线程列表
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return coord, threads