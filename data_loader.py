import pickle
import numpy as np
import os
import nibabel as nib

training_data_path = "data\\MICCAI-2013-SATA-Challenge-Data-Std-Reg\\diencephalon\\training"
testing_data_path = "data\\MICCAI-2013-SATA-Challenge-Data-Std-Reg\\diencephalon\\testing"
val_ratio = 0.3        # 验证数据集使用比例
seed = 100             # 随机种子
preserving_ratio = 0.1 # filter out 2d images containing < 10% non-zeros

"""
17-40行代码的目的为获取训练集 验证集 测试集的压缩文件
"""

f_train_all = os.listdir(training_data_path)
train_all_num = len(f_train_all)         # 训练集所使用的全部数据集
val_num = int(train_all_num * val_ratio) # 验证集数量

f_train = []                            # 训练集
f_val = []                              # 验证集

val_idex = np.random.randint(0,train_all_num - 1,size=val_num)  # 验证集索引
for i in range(train_all_num):  #将验证集与训练集分为两部分
    if i in val_idex:
        f_val.append(f_train_all[i])
    else:
        f_train.append(f_train_all[i])

f_test = os.listdir(testing_data_path) # 加载测试集

train_3d_num, val_3d_num, test_3d_num = len(f_train), len(f_val), len(f_test)  # 训练集 验证集 测试集数量
#train_3d_num, val_3d_num  = len(f_train), len(f_val)

X_train = []
for fi, f in enumerate(f_train):  # fi为训练集索引 f为训练集的文件名
    print("processing [{}/{}] 3d image ({}) for training set ...".format(fi + 1, train_3d_num, f))
    img_path = os.path.join(training_data_path, f)  # 相当于连接字符串 局部变量
    img = nib.load(img_path).get_data()            # 获取数据
    img_3d_max = np.max(img)                       # 求出一张3D图像中的最大像素值
    img = img / img_3d_max * 255                   # 像素值调整至0~255
    for i in range(img.shape[2]):
        img_2d = img[:, :, i]
        # filter out 2d images containing < 10% non-zeros
        if float(np.count_nonzero(img_2d)) / img_2d.size >= preserving_ratio:
            img_2d = img_2d / 127.5 - 1            # 数据映射至-1~1 先当与数据预处理操作
            img_2d = np.transpose(img_2d, (1, 0))  # 转置 将图片摆正
            X_train.append(img_2d)                 # 加入训练集


X_val = []
for fi, f in enumerate(f_val):
    print("processing [{}/{}] 3d image ({}) for validation set ...".format(fi + 1, val_3d_num, f))
    img_path = os.path.join(training_data_path, f)
    img = nib.load(img_path).get_data() # 获取数据
    img_3d_max = np.max(img)            # 求出一张3D图像中的最大像素值
    img = img / img_3d_max * 255        #
    for i in range(img.shape[2]):       #
        img_2d = img[:, :, i]           # 2D图像
        # filter out 2d images containing < 10% non-zeros
        if float(np.count_nonzero(img_2d)) / img_2d.size >= preserving_ratio:
            img_2d = img_2d / 127.5 - 1  #
            img_2d = np.transpose(img_2d, (1, 0))
            X_val.append(img_2d)

X_test = []
for fi, f in enumerate(f_test):
    print("processing [{}/{}] 3d image ({}) for test set ...".format(fi + 1, test_3d_num, f))
    img_path = os.path.join(testing_data_path, f)
    img = nib.load(img_path).get_data()
    img_3d_max = np.max(img)
    img = img / img_3d_max * 255
    for i in range(img.shape[2]):
        img_2d = img[:, :, i]  # 取一个正面切片
        # filter out 2d images containing < 10% non-zeros
        if float(np.count_nonzero(img_2d)) / img_2d.size >= preserving_ratio:
            img_2d = img_2d / 127.5 - 1
            img_2d = np.transpose(img_2d, (1, 0))
            X_test.append(img_2d)

X_train = np.asarray(X_train)
X_train = X_train[:, :, :, np.newaxis]
X_val = np.asarray(X_val)
X_val = X_val[:, :, :, np.newaxis]
X_test = np.asarray(X_test)
X_test = X_test[:, :, :, np.newaxis]

# save data into pickle format
data_saving_path = 'data/MICCAI13_SegChallenge/'
if not (os.path.exists(data_saving_path)):
    os.mkdir(data_saving_path)

print("save training set into pickle format")
with open(os.path.join(data_saving_path, 'training.pickle'), 'wb') as f:
    pickle.dump(X_train, f, protocol=4)

print("save validation set into pickle format")
with open(os.path.join(data_saving_path, 'validation.pickle'), 'wb') as f:
    pickle.dump(X_val, f, protocol=4)

print("save test set into pickle format")
with open(os.path.join(data_saving_path, 'testing.pickle'), 'wb') as f:
    pickle.dump(X_test, f, protocol=4)
print("processing data finished!")
