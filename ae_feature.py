# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         _base
# Description:  生成样本特征，并存为Excel文件
# Author:       王洪磊
# Email：       wang_hl007@163.com
# Date:         2021/6/27
# -------------------------------------------------------------------------------

import h5py
import pywt
import librosa
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib2 import Path
from scipy.fftpack import dct
import warnings

warnings.filterwarnings("ignore")
#
# def lcc_norm(data, cep_num, filter_bans, frame_length, hop_length):
#     """
#
#     :param data:
#     :param cep_num:
#     :param filter_bans:
#     :param frame_length:
#     :param hop_length:
#     :return:
#
#     cep_num =13
#
#     """
#     # 分帧
#     eps = 1e-1000
#     signal_length = len(data)
#     frame_shift = frame_length - hop_length  # 帧移
#     num_frames = np.ceil(signal_length / frame_shift)
#
#     pad_zero = np.zeros(int(num_frames * frame_shift - signal_length))
#     pad_data = np.append(data, pad_zero).reshape(-1, frame_shift)
#     hop_data = np.zeros((pad_data.shape[0], hop_length))
#     hop_data[:-1, :] = pad_data[1:, :hop_length]
#
#     signal = np.concatenate((pad_data, hop_data), axis=1)
#
#     # 加汉明窗
#     hammming = np.hamming(frame_length)
#     signal = signal * hammming
#
#     # 傅里叶变换频谱和能量谱
#     mag_frames = np.absolute(np.fft.rfft(signal, axis=1))[:, 1:]
#     pow_frames = np.array((1.0 / len(mag_frames)) * (mag_frames ** 2))  # Power Spectrum
#
#     # Band-Pass Filter
#     filter_pow = np.dot(pow_frames, filter_bans)
#     filter_pow[filter_pow < eps] = eps
#     log_pow = np.log(filter_pow)
#
#     lcc = dct(log_pow, n=cep_num)
#     return lcc.mean(axis=0)
#
#
# def get_filter(frame_length, fs_KHz, begin, band, filter_num, mod='linear', ):
#     bands_point = np.arange(0, fs_KHz, fs_KHz / frame_length * 2)
#     filter_b = np.zeros((int(frame_length / 2), filter_num))
#     if mod == 'linear':
#         for i in range(filter_num):
#             start_band = begin + i * band
#             end_band = begin + (i + 2) * band
#
#             pos = bands_point[(start_band < bands_point) & (bands_point < end_band)]
#             start_pos = int(np.argwhere(bands_point == pos[0])[0]) - 1
#             end_pos = int(np.argwhere(bands_point == pos[-1])[0]) + 1
#
#             filter_b[start_pos:end_pos, i] = 1
#     return filter_b
#


def ae_data(AE_group, channels, start=0, end=10):
    """
    读取声发射数据
    :param AE_group:
    :param channels: 通道名称
    :param start: 开始时间  单位 ms
    :param end:  结束时间  单位 ms
    :return:
    """

    channels_data = pd.DataFrame(columns=channels, dtype=np.float)
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


class Feature(object):

    # FEATURE = ['MFCC', 'LCC', 'ZCR', 'SC']

    def __init__(self, data=None, fs=3e6, frame_length=2, hop_length=1,
                 mfcc_num=13, lcc_num=13,   # MFCC LCC个数
                 thresh=0,  # 过零率门槛值
                 ):
        """

        :param data:
        :param fs: 采样频率
        :param frame_length: 帧长
        :param hop_length:  相邻帧重叠长度
        :param mfcc_num:  梅尔倒谱系数个数
        :param lcc_num: 线性倒谱系数个数
        :param thresh: 过零率的阈值
        """
        # 初始化类能做 特征方法名与特征方法的字典
        self.FEATURE = {'MFCC': self.mfcc, 'LCC': self.lcc_norm,
                        'ZCR': self.zero_cross_rate, 'SC': self.spectral_centroid}
        self.feature = {}
        self.filter = None
        self.mfcc_num = mfcc_num
        self.lcc_num = lcc_num
        self.thresh = thresh
        self.fs = int(fs)  # 单位 KHz
        self.frame_length = int(frame_length*self.fs/1000)
        self.hop_length = int(hop_length*self.fs/1000)
        self.feature_dict = {'MFCC': self.mfcc}   # 方法名与特征方法的字典
        self.set_filter()
        if data is None:
            self.data = None
        else:
            self.set_data(data)

    def set(self, **kwargs):
        args = {'fs': self.fs, 'frame_length': self.frame_length, 'hop_length': self.hop_length,
                'mfcc_num': self.hop_length, 'lcc_num': self.hop_length, 'thresh':self.hop_length}
        for arg in args:
            if arg in kwargs:
                args[arg] = kwargs[arg]

    def set_data(self, data):
        if isinstance(data, np.ndarray):
            self.data = data.flatten()
        elif isinstance(data, pd.Series):
            data = data.values
            self.data = data.flatten()
        else:
            raise TypeError('输入数据必须是numpy数组')

    def zero_cross_rate(self, data=None):
        if data is not None:
            self.set_data(data)

        data = self.data
        data = data - data.mean() - self.thresh
        zcr = librosa.feature.zero_crossing_rate(data, frame_length=self.frame_length,
                                                 hop_length=self.hop_length)[0].mean()

        return {'ZCR': zcr}

    def mfcc(self, data=None):
        if data is not None:
            self.set_data(data)

        data = self.data
        mfcc_data = librosa.feature.mfcc(data, sr=self.fs, n_mfcc=self.mfcc_num, n_fft=self.frame_length,
                                         hop_length=self.hop_length).mean(axis=1)
        mfcc_name = ['MFCC-'+str(i+1) for i in range(self.mfcc_num)]
        mfcc_data = dict(zip(mfcc_name, mfcc_data))
        return mfcc_data

    def spectral_centroid(self, data=None):
        if data is not None:
            self.set_data(data)

        data = self.data
        sc = librosa.feature.spectral_centroid(data, sr=self.fs, hop_length=self.hop_length)[0].mean()
        return {'SC': sc}

    def set_filter(self, begin=0, band=10e3, filter_num=100, mod='linear'):
        bands_point = np.arange(0, self.fs, self.fs / self.frame_length * 2)
        filter_band = np.zeros((int(self.frame_length / 2), filter_num))
        if mod == 'linear':
            for i in range(filter_num):
                start_band = begin + i * band
                end_band = begin + (i + 2) * band

                pos = bands_point[(start_band < bands_point) & (bands_point < end_band)]
                start_pos = int(np.argwhere(bands_point == pos[0])[0]) - 1
                end_pos = int(np.argwhere(bands_point == pos[-1])[0]) + 1

                filter_band[start_pos:end_pos, i] = 1
        self.filter = filter_band

    def lcc_norm(self, data=None):
        """
        计算普通倒谱系数，滤波器组的频率和帧相关

        :param data:
        # :param cep_num:
        :return:
        """
        if data is not None:
            self.set_data(data)
        data = self.data

        # 分帧
        eps = 1e-1000
        signal_length = len(data)
        frame_shift = self.frame_length - self.hop_length  # 帧移
        num_frames = np.ceil(signal_length / frame_shift)

        pad_zero = np.zeros(int(num_frames * frame_shift - signal_length))
        pad_data = np.append(data, pad_zero).reshape(-1, frame_shift)
        hop_data = np.zeros((pad_data.shape[0], self.hop_length))
        hop_data[:-1, :] = pad_data[1:, :self.hop_length]

        signal = np.concatenate((pad_data, hop_data), axis=1)

        # 加汉明窗
        hammming = np.hamming(self.frame_length)
        signal = signal * hammming

        # 傅里叶变换频谱和能量谱
        mag_frames = np.absolute(np.fft.rfft(signal, axis=1))[:, 1:]
        pow_frames = np.array((1.0 / len(mag_frames)) * (mag_frames ** 2))  # Power Spectrum

        # Band-Pass Filter
        filter_pow = np.dot(pow_frames, self.filter)
        filter_pow[filter_pow < eps] = eps
        log_pow = np.log(filter_pow)

        lcc = dct(log_pow, n=self.lcc_num)
        lcc = lcc.mean(axis=0)
        lcc_names = ['LCC-{}'.format(i+1) for i in range(self.lcc_num)]
        return dict(zip(lcc_names, lcc))

    def denoise(self, data=None, thresh=1, wavelet='db4',  mod='softy'):
        if data is not None:
            self.set_data(data)
        data = self.data

        coeff = pywt.wavedec(data, wavelet, mode="per")
        if mod == 'softy':
            mad = np.mean(np.absolute(coeff[-thresh] - np.mean(coeff[-thresh])))
            sigma = (1 / 0.6745) * mad
            thresh = sigma * np.sqrt(2 * np.log(len(data)))
            coeff[1:] = (pywt.threshold(i, value=thresh, mode='hard') for i in coeff[1:])

        else:
            coeff[1:] = (pywt.threshold(i, value=thresh, mode='hard') for i in coeff[1:])

        return pywt.waverec(coeff, wavelet, mode='per')

    def set_feature_names(self, names):
        feat_names = {}
        for name in names:
            if name in self.FEATURE:
                feat_names[name] = self.FEATURE[name]
            else:
                print('不支持计算特征{}'.format(name))
        if len(feat_names) > 0:
            self.feature_dict = feat_names
        else:
            raise NameError('仅支持特征：{}'.format(self.FEATURE))

    def get_feature(self, data=None, names=None):
        if names is not None:
            self.set_feature_names(names)
        if data is not None:
            self.set_data(data)

        self.feature = {}
        for fun in self.feature_dict.values():
            self.feature.update(fun())
            # feat_i = fun()
            # for k, v in feat_i.items():
            #     feature[k] = v
        return self.feature


def feature(AE_group, channels, ae_feature, sample_time):

    total_time = AE_group[channels[0]].shape[0]
    sample_num = int(np.ceil(total_time/sample_time))  # 声发射数据总数

    sample = pd.DataFrame([])
    for i in tqdm(range(sample_num)):

        start = i*sample_time
        end = (i+1)*sample_time
        data = ae_data(AE_group, channels, start, end)  # 一个试样的声发射数据

        sample_i = data.apply(lambda x: pd.Series(ae_feature.get_feature(x)))
        sample_i['sample_num'] = i
        sample = sample.append(sample_i)
        # # MFCC
        # mfcc = data.apply(feature.mfcc)
        # mfcc.index = ['MFCC-'+str(i+1) for i in range(len(mfcc))]
        # sample_i = sample_i.append(mfcc)
        #
        # # LCC
        # lcc = data.apply(feature.lcc_norm)
        # lcc.index = ['LCC-' + str(i+1) for i in range(len(lcc))]
        # sample_i = sample_i.append(lcc)

        # # ZCR
        # zcr = pd.DataFrame(data.apply(feature.zero_cross_rate)).T
        # zcr.index = ['ZCR']
        # sample_i = sample_i.append(zcr)
        #
        # # SC
        # sc = pd.DataFrame(data.apply(feature.spectral_centroid)).T
        # sc.index = ['SC']
        # sample_i = sample_i.append(sc)
        #
        # sample_i['sample_num'] = i
        # sample = sample.append(sample_i)

    # 规整特征形状,
    names = [i for i in sample.columns if 'CH' in i]
    keys = [i for i in sample.index.unique()]
    regular_sample = pd.DataFrame()
    for name in names:
        sample_i = sample.loc[:, [name, 'sample_num']]
        time = sample['sample_num'].unique()*sample_time
        regular_sample_i = pd.DataFrame(time,  columns=['time'])
        for key_i in keys:

            feat_i = sample_i.loc[key_i,]
            feat_i = feat_i.sort_values(by=['sample_num'])
            regular_sample_i.loc[:, key_i] = feat_i[name].values
        regular_sample_i['Channel'] = name
        regular_sample = regular_sample.append(regular_sample_i)

    return regular_sample


def get_sample(h5_file, channels, ae_feature, sample_time=40, target=10e3):
    """
     target ms后实验不会坏为安全样本，标记为正样本1，否则为危险样本标记为-1
    :param h5_file:
    :param channels:
    :param feature:
    :param sample_time:
    :param target:
    :return:
    """
    AE_group = h5_file['AE']
    has_channels = [i for i in AE_group.keys() if 'max_min' not in i]
    channels = [i for i in channels if i in has_channels]
    print('使用的通道：', channels)

    sample = feature(AE_group, channels, ae_feature, sample_time=sample_time)

    load = load_data(h5_file)
    critical_time = load.iloc[load['应力(MPa)'].argmax(), 0] * 1000

    def map_label(x):
        if x < critical_time - target:
            return 1
        else:
            return -1
    sample['label'] = sample['time'].apply(map_label)

    return sample


def main():
    channels = ['CH1', 'CH2', 'CH5', 'CH6', 'CH9', 'CH10', 'CH13']
    save_folder = Path(r'D:\WPS云文件\代价敏感预测煤岩破坏\数据')
    folder = Path(r'E:\声发射数据与分析\北科大单轴实验\实验结果')
    ae_feature = Feature()
    ae_feature.set_feature_names(['MFCC', 'LCC', 'ZCR', 'SC'])

    paths = [i for i in folder.iterdir()]
    for path in paths[-7:-5]:
        print(path)
        h5_file = h5py.File(path, mode='r')
        sample = get_sample(h5_file, channels, ae_feature)
        h5_file.close()

        excel_name = path.with_suffix('.xlsx')
        sample.to_excel(save_folder/excel_name.name)




