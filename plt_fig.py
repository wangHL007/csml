#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2021/7/10 9:26
# Author  : 王洪磊
# File : plt_fig.py 

import h5py
import pandas as pd
import numpy as np
from pathlib2 import Path
import matplotlib.pyplot as plt
import warnings
import matplotlib as mpl

font = {'family': 'Times New Roman', 'size': 6}
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 6
warnings.filterwarnings("ignore")


def ae_data(AE_group, channels, start=0, end=10):
    """
    读取声发射数据
    :param AE_group:
    :param channels: 通道名称
    :param start: 开始时间  单位 ms
    :param end:  结束时间  单位 ms

    :return:
    """

    channels_data = pd.DataFrame(columns=channels)
    for channel in channels:
        ae_dataset = AE_group[channel]
        channel_data = ae_dataset[start:end, :].flatten()
        channels_data[channel] = channel_data.astype(np.float)

    channels_data = (channels_data - 2**15) / 2**15 * 10

    return channels_data


def load_data(h5_file, start=None, end=1000):
    """
    # 读取加载实验数据
    :param h5_file:
    :param start: 开始时间  单位 ms, 当start=None时返回全部的加载数据
    :param end:  结束时间  单位 ms
    :return:
    """

    load_dataset = h5_file['load']
    load_keys = ['时间(s)', '横向变形(mm)', '应力(MPa)', '应变(%)']

    load = pd.DataFrame(load_dataset.value, columns=load_keys)
    if start is None:
        data = load
    else:
        start, end = start / 1000, end / 1000
        data = load[(start < load['时间(s)']) & (load['时间(s)'] < end)]
    return data


def valid_channel(AE_group, channels):
    """

    :param AE_group:
    :param channels:
    :return: 如果AE_group中含有输入的通道，则这些通道名称，否则返回AE_group中第一个通道名称
    """
    has_channels = [i for i in AE_group.keys() if 'max_min' not in i]
    channels = [i for i in channels if i in has_channels]

    if len(channels) == 0:
        channels = has_channels[0]

    print('使用的通道：', channels)
    return channels


def wave_fig(path, channel, delay_time=0, save_path=None):
    # fig_size = np.array([12.5, 6.5])
    # rect1 = 0.1, 0.15, 0.78, 0.8  # [12.5, 6.5]
    fig_size = np.array([7, 4])
    rect1 = 0.13, 0.17, 0.74, 0.76  # [7, 4]
    size = fig_size / 2.54

    x_label = 'Time /s'
    y_label1 = 'Stress /MPa'
    y_label2 = 'Amplitude /V'

    fig = plt.figure(figsize=size)
    ax_load = fig.add_axes(rect1, label='load')
    ax_load.set_xlabel(x_label, fontdict=font)
    ax_load.set_ylabel(y_label1, fontdict=font)
    ax_ae = ax_load.twinx()
    ax_ae.set_ylabel(y_label2, fontdict=font)
    ax_ae.tick_params(direction='in', width=0.5, length=2)
    ax_load.tick_params(direction='in', width=0.5, length=2)

    spines = ax_load.spines
    for spine in spines:
        spines[spine].set_linewidth(0.5)

    spines = ax_ae.spines
    for spine in spines:
        spines[spine].set_linewidth(0.5)

    h5_file = h5py.File(path, mode='r')
    AE_group = h5_file['AE']
    channel = valid_channel(AE_group, channel)  # 验证文件中有输入的通道

    load = load_data(h5_file)
    ax_load.plot(load['时间(s)'], load['应力(MPa)'], linewidth=0.5, color='#0000FF', label='Stress')

    ae_dataset = AE_group[channel[0] + '-max_min']
    ae = ae_dataset[2:-1].astype(np.float)
    ae = (ae - 2**15) / 2**15 * 10
    diff = abs(ae[:, 0]) - abs(ae[:, 1])
    ae = ae[diff < 0.1, :]
    ae = ae.flatten()

    ae_time = np.linspace(0, ae.size/2000, ae.size) - delay_time
    ax_ae.plot(ae_time, ae, linewidth=0.5, color='r', label='AE')

    if save_path is not None:
        fig.savefig(save_path, dpi=600)
    plt.close(fig)


def feat_fig():

    # fig_size = np.array([12.5, 6.5])
    # rect1 = 0.1, 0.15, 0.78, 0.8  # [12.5, 6.5]
    def figure(load=None):
        fig_size = np.array([4.5, 3.5])
        rect1 = 0.19, 0.2, 0.64, 0.76  # [7, 4]
        size = fig_size / 2.54

        x_label = 'Time /s'
        y_label1 = 'Stress /MPa'

        fig = plt.figure(figsize=size)
        ax1 = fig.add_axes(rect1, label='load')
        ax1.set_xlabel(x_label, fontdict=font)
        ax1.set_ylabel(y_label1, fontdict=font)

        ax2 = ax1.twinx()
        ax2.tick_params(direction='in', width=0.5, length=2)
        ax1.tick_params(direction='in', width=0.5, length=2)
        if load is not None:
            ax1.plot(load['时间(s)'], load['应力(MPa)'], linewidth=0.5, color='k', label='Stress')

        return fig

    load_folder = Path(r'D:\WPS云文件\代价敏感预测煤岩破坏\数据\应力数据')
    save_folder = Path(r'D:\WPS云文件\代价敏感预测煤岩破坏\数据\特征图')
    folder = Path(r'D:\WPS云文件\代价敏感预测煤岩破坏\数据\100ms')
    # names = ['LCC-' + str(i + 1) for i in range(6)]
    names = ['MFCC-' + str(i + 1) for i in range(6)]
    # names = ['SC', 'ZCR']
    for path in folder.iterdir():
        print('now plot {}'.format(path.name))
        data = pd.read_excel(path)
        time = data['time']/1000
        load_path = load_folder / path.name
        load = pd.read_excel(load_path)

        for name in names:
            print(name)
            sample_name = name + path.name
            save_path = save_folder/sample_name
            feat_data = data[name]
            feat_data = feat_data.rolling(10).mean()
            color = data['label'].map({1: '#0000FF', -1: '#DC143C'})
            fig = figure(load)
            ax_feat = fig.axes[1]
            ax_feat.scatter(time, feat_data, s=0.1, c=color)
            ax_feat.set_ylabel(name, fontdict=font)
            fig.savefig(save_path.with_suffix('.png'), dpi=600)

            plt.close(fig)

    # ax_feat.plot(time, data['MFCC-1'], linewidth=0.5, color='r', label='AE')


def main():
    channel = ['CH13']
    save_folder = Path(r'D:\WPS云文件\代价敏感预测煤岩破坏\数据\应力-波形图')
    folder = Path(r'F:\声发射数据与分析\北科大单轴实验\实验结果')

    delay_time = {'宽沟3-1号煤样.hdf5': 0, '宽沟3-2号煤样.hdf5': 14.1,
                  '宽沟7-1号煤样.hdf5': 5.6, '宽沟7-2号煤样.hdf5': 5.1,
                  '宽沟11-1号煤样.hdf5': 0, '宽沟11-3号煤样.hdf5': 2,
                  '宽沟15-1号煤样.hdf5': 0, '宽沟15-2号煤样.hdf5': 0.3,
                  '宽沟20-1号煤样.hdf5': 0, '宽沟20-3号煤样.hdf5': 0.3,
                  }

    paths = [i for i in folder.iterdir()]

    for path in paths[-10:]:
        # path = paths[-7]
        if path.suffix == '.hdf5':
            sample_name = channel[0] + path.name
            save_path = save_folder/sample_name
            print('正在处理===》 {} '.format(path.name))
            wave_fig(path, channel, delay_time=delay_time[path.name], save_path=save_path.with_suffix('.png'))
        else:
            print('{} 文件名不是hdf5'.format(path.name))

