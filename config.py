from easydict import EasyDict as edict
import json
import os

config = edict()
config.TRAIN = edict()

config.TRAIN.batch_size = 25   # 小批量数据的大小
config.TRAIN.early_stopping_num = 10    #
config.TRAIN.lr = 0.0001  # 学习率
config.TRAIN.lr_decay = 0.5  # 降低学习率
config.TRAIN.decay_every = 5  # 每五次迭代降低一次学习率
config.TRAIN.beta1 = 0.5  # beta1 in Adam optimiser
config.TRAIN.n_epoch = 9999  # 迭代次数？
config.TRAIN.sample_size = 50  # 测试所使用的图片数量
config.TRAIN.g_alpha = 15  # weight for pixel loss
config.TRAIN.g_gamma = 0.0025  # weight for perceptual loss
config.TRAIN.g_beta = 0.1  # weight for frequency loss
config.TRAIN.g_adv = 1  # weight for frequency loss

config.TRAIN.seed = 100  # 随机种子
config.TRAIN.epsilon = 0.000001


config.TRAIN.VGG16_path = os.path.join('trained_model', 'VGG16', 'vgg16_weights.npz')  # VGG模型的位置
config.TRAIN.training_data_path = os.path.join('data', 'MICCAI13_SegChallenge', 'training.pickle')  # 训练集
config.TRAIN.val_data_path = os.path.join('data', 'MICCAI13_SegChallenge', 'validation.pickle')     # 验证集
config.TRAIN.testing_data_path = os.path.join('data', 'MICCAI13_SegChallenge', 'testing.pickle')    # 测试集
config.TRAIN.mask_Gaussian1D_path = os.path.join('mask', 'Gaussian1D')
config.TRAIN.mask_Gaussian2D_path = os.path.join('mask', 'Gaussian2D')
config.TRAIN.mask_Poisson2D_path = os.path.join('mask', 'Poisson2D')

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")