# -*- coding: utf-8 -*-
# 存放全局变量
# 主要用来控制生成存放文件和结果的目录
# 并统一生成路径等
import os

# mnist图片下载目录路径
global mnist_dir
mnist_dir = './asset/data/mnist'


# 存放中间文件的目录
global midfile_dir
midfile_dir = './asset/data/midfile/'
if os.path.exists(midfile_dir):
	os.makedirs(midfile_dir)

# 文件前缀名
global mid_prefix
mid_prefix = "ac_gan"

# 生成的图片文件目录
global gen_image_dir
gen_image_dir = "./asset/data/image/"
if os.path.exists(gen_image_dir):
	os.makedirs(gen_image_dir)
