# -*-coding:utf-8-*-
"""
该脚本用于训练模型
该模型使用经典的VGG结构
"""
from argparse import ArgumentParser
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model
from keras.optimizers import SGD
from config import PATIENCE, TRAIN_SAMPLES, VALID_SAMPLES, EPOCH
from utils import get_available_gpus, get_available_cpus
from model import build_model
from data_generator import DataGenSequence, get_data_all
from loss_function import categorical_crossentropy_color

# 配置GPU环境
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os
# 进行配置，每个GPU使用80%上限显存
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)
KTF.set_session(session)


def parse_command_params():
    """
    命令行参数解析器
    :return:
    """
    ap = ArgumentParser()  # 创建解析器
    ap.add_argument('-p', '--pretrained', help="path of your model files expected .h5 or .hdf5")
    ap.add_argument('-s', '--show', default='no', help="if you want to visualize training process")
    ap.add_argument('-m', '--method', default='all', help='fit all or fit generator')
    ap.add_argument('-b', '--batch', default=64, help='batch size of training')
    ap.add_argument('-e', '--epochs', default=100, help='how many epochs to train')
    ap.add_argument('-l', '--loss', default='ce', help='which loss function to train')
    args_ = vars(ap.parse_args())
    return args_


def get_model(pretrained):
    """
    按照机器配置构建合理模型
    :return:
    """
    if get_available_gpus() > 1:
        model = build_model()
        print(pretrained)
        if pretrained:
            my_model.load_weights(pretrained)
        else:
            pass
        model = multi_gpu_model(model, get_available_gpus())

    else:
        model = build_model()
        if pretrained:
            model.load_weights(pretrained)

    return model


def get_callbacks():
    """
    设置一些回调
    :return:
    """
    model_checkpoint = ModelCheckpoint('../models/training_best_weights.h5',
                                       monitor='val_loss', verbose=True, save_best_only=True,
                                       save_weights_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE)
    reduce_lr = ReduceLROnPlateau(monitor='lr', factor=0.1, patience=PATIENCE // 4, verbose=True)

    callbacks_ = [model_checkpoint, early_stopping, reduce_lr]
    return callbacks_


def train(args_, model):
    """
    训练
    :param args_: 命令行参数
    :param model: 模型
    :return:
    """

    optimizer = SGD(lr=1e-2, momentum=0.9, nesterov=True, clipnorm=0.5)

    if args_['loss'] == 'ce':
        model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    elif args_['loss'] == 'vce':
        model.compile(optimizer=optimizer, loss=categorical_crossentropy_color)
    else:
        print("Please enter valid loss function")
        return None

    if args_['method'] == 'all':
        x_all, y_all = get_data_all()
        model.fit(x_all, y_all,
                  batch_size=int(args_['batch']),
                  epochs=int(args_['epochs']),
                  validation_split=0.2,
                  verbose=(args_['show'] == 'yes'),
                  callbacks=get_callbacks(),
                  shuffle=True)
    else:
        model.fit_generator(DataGenSequence('train', int(args_['batch'])),
                            steps_per_epoch=TRAIN_SAMPLES // int(args_['batch']),
                            validation_data=DataGenSequence('valid', int(args_['batch'])),
                            validation_steps=VALID_SAMPLES // int(args_['batch']),
                            epochs=EPOCH,
                            verbose=(args_['show'] == 'yes'),
                            callbacks=get_callbacks(),
                            )
    if not os.path.exists('../models'):
        os.mkdir('../models')
    model.save_weights('../models/my_model_weights.h5')


if __name__ == '__main__':
    args = parse_command_params()
    my_model = get_model(args['pretrained'])
    print(my_model.summary())
    train(args, my_model)
