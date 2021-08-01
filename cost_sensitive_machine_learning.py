# !/usr/bin/python3
# -*-coding:utf-8-*-
# Author: 王洪磊
# Email: wang_hl007@163.com
# CreatDate: 2021/7/27 20:57
# Description:
import copy
import numpy as np
import pandas as pd
from pathlib2 import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import matplotlib as mpl
import xlwings

font = {'family': 'Times New Roman', 'size': 8}
mpl.rcParams['font.size'] = 8
mpl.rcParams['font.sans-serif'] = ['SimSun']
mpl.rcParams['axes.unicode_minus'] = False

class CostSensitiveMachineLearning(object):
	# TODO 设置分类的class weight参数
	# TODO 适配不同的基学习器
	# TODO 基学习器参数适配
	# 支持的基学习器
	BASE_CLASSIFIER_CLASS = dict(决策树=DecisionTreeClassifier,
	                             逻辑回归=LogisticRegression,
	                             支持向量机=SVC)
	
	def __init__(self, x_train: pd.DataFrame, y_train: pd.Series,x_test: pd.DataFrame, y_test: pd.Series,
	              plans: np.ndarray ,base_classifier_para: dict, base_classifier_name: str = '决策树'):
		"""
		train_test_index: 关键字为train_index和test_index的字典，值为列表其中每个元素均在样本特征的索引，
		样本训练集和测试集的划分，
		:param x: 样本特征
		:param y:两分类的样本标签， 0是安全，1是危险。 y的索引必须与x的索引一致, 值较小的为安全
		# :param train_test_index: 关键字为train_index和test_index的字典，值为列表其中每个元素均在样本特征的索引，样本训练集和测试集的划分，
		:param plans: 防治方案的成本与效果，第一列为自然状态为安全的损失，第二列为自然状态危险的损失
		:param base_classifier_para: 基学习器的参数
		:param base_classifier_name: 基学习器的名称

		"""
		self.x_train = x_train
		self.x_test = x_test
		
		label = y_train.unique()
		self.y_trans = {label[0]: 0, label[1]: 1}
		self.y_train = y_train.map(self.y_trans)
		self.y_test = y_test.map(self.y_trans)
		
		self.x = None
		self.y = None

		self.train_index = None
		self.test_index = None
		self.cost = None

		self.plans_init = np.array(plans)  # 防治方案的成本与效果，第一列为自然状态为安全的损失，第二列为自然状态危险的损失
		self.plans_selected = None
		self.plans_thresholds = None
		
		self.base_classifier_name = base_classifier_name
		self.base_classifier_paras = base_classifier_para
		self.base_classifier_class = self.BASE_CLASSIFIER_CLASS.get(self.base_classifier_name)
		self.base_classifier = None
		
		self.class_weight = None
		self.model = []  # 代价敏感分类器集合
		self.y_pred = np.array([])  # 分类器预测结果
		self.predict_result = np.array([])  # 模型预测结果
		self.is_init = False
		self.is_fitted = False
		
		
		self.set_dataset()
		# self.set_train_test_index(train_test_index)
		self._init_model()
		
	def set_dataset(self, x=None, y:pd.Series=None):
		# 0是安全，1是危险
		if x is None:
			self.x = self.x_train.append(self.x_test)
			self.y = self.y_train.append(self.y_test)
		else:
			self.x = x
			label = y.unique()
			self.y = y.map({label[0]: 1, label[1]: 0})
		
	def set_train_test_index(self, index:dict):
		self.train_index, self.test_index = index.get('train_index'), index.get('test_index')
		
		self._set_train_data()
		self._set_test_data()
		
	def _set_train_data(self, index=None):
		if index is None:
			self.x_train = self.x.loc[self.train_index, :]
			self.y_train = self.y.loc[self.train_index]
		else:
			self.x_train = self.x.loc[index, :]
			self.y_train = self.y.loc[index]
	
	def _set_test_data(self, index=None):
		if index is None:
			self.x_test = self.x.loc[self.test_index, :]
			self.y_test = self.y.loc[self.test_index]
		else:
			self.x_test = self.x.loc[index, :]
			self.y_test = self.y.loc[index]
	
	def _init_model(self):
		
		plans_selected = self.select_plans()   # 优化防治方案
		if plans_selected is None:
			print('优化方案失败')
			return False
		
		class_weight = self._init_class_weight()  # 计算各分类器类别权重
		if class_weight is None:
			print('初始化样本权重失败')
			return False
		
		self._update_model()
		return True
		
	# Optimization and sequencing of schemes
	def select_plans(self, plans=None):
		if plans is None:
			plans = np.array(self.plans_init)
		
		plans_opt = [plans[0, :]]
		thresholds = [0]
		remain_plans = plans[1:]
		while remain_plans.shape[0] > 0:
			elect_result = self._elect(plans_opt[-1], thresholds[-1], remain_plans)
			plans_opt.append(elect_result[0])
			thresholds.append(elect_result[1])
			remain_plans = plans[plans[:, 1] < plans_opt[-1][1]]
			print('now opt plans', plans_opt[-1])
			print('now opt threshold', thresholds[-1])
		
		if len(plans_opt) < 2:
			print('最终优化方案个数小于2')
			return
		else:
			self.plans_selected, self.plans_thresholds = np.array(plans_opt), np.array(thresholds)
			return self.plans_selected
	
	@staticmethod
	def _elect(now_opt_plan,  thresh, plans:np.ndarray):
		thresholds = []
		for plan in plans:
			print(plan)
			threshold = (plan[0] - now_opt_plan[0]) / (now_opt_plan[1] - now_opt_plan[0] - (plan[1] - plan[0]))
			thresholds.append(threshold)
		thresholds = np.array(thresholds)
		thresholds = thresholds[thresholds > thresh]
		# 找到交点最小的方案
		min_thresh = thresholds.min()
		min_wheres = np.argwhere(thresholds == min_thresh).flatten()
		min_plans = plans[min_wheres, :]
		print(min_plans)
		min_cost = min_plans[min_plans[:, 1].argmin()]
		elect_result = [min_cost, min_thresh]
		
		
		return elect_result
	
	def _init_class_weight(self):

		class_weights = []
		for i in range(1, self.plans_selected.shape[0]):
			plan1 = self.plans_selected[i - 1]
			plan2 = self.plans_selected[i]
			class_weight = -(plan1[0]-plan2[0])/(plan1[1]-plan2[1])
			
			class_weights.append({0: class_weight, 1: 1})
		
		if len(class_weights) > 0:
			self.class_weight = class_weights
			return self.class_weight
	
	def _update_model(self):
		for class_weight in self.class_weight:
			base_classifier_para = self.base_classifier_paras
			base_classifier_para['class_weight'] =  class_weight
			print(base_classifier_para)
			self.model.append(self.base_classifier_class(**base_classifier_para))
			self.is_init = True
		return True

	def set_plans(self, plans, update_model=True):
		raw_plans_init = copy.deepcopy(self.plans_init)
		self.plans_init = plans
		if update_model:
			set_success = self._init_model()
			if not set_success:
				print('设置基学习器参数失败')
				self.plans_init = raw_plans_init
	
	def set_class_weight(self, class_weight, update_model=True):
		raw_class_weight = copy.deepcopy(self.class_weight)
		self.class_weight = class_weight
		if update_model:
			set_success = self._update_model()
			if not set_success:
				print('设置基学习器参数失败')
				self.class_weight = raw_class_weight
			
	def set_base_classifier_paras(self, para, update_model=True):
		raw_base_classifier_paras = copy.deepcopy(self.base_classifier_paras)
		self.base_classifier_paras = para
		if update_model:
			set_success = self._update_model()
			if not set_success:
				print('设置基学习器参数失败')
				self.base_classifier_paras = raw_base_classifier_paras

	def fit(self, x=None, y=None):
		if x is None:
			x = self.x_train
			y = self.y_train
		# print(y.unique())
		
		
		for clf in self.model:
			if y.unique().shape[0] < 2:
				print('训练集中只有1类标签')
				continue
			print('正在训练...', clf)
			clf.fit(x, y)

		self.is_fitted = True
		
	
	def predict(self, x=None):
		if not self.is_fitted:
			self.fit()

		if x is None:  # TODO 检查输入x是否与self.x列名一致
			x = self.x_test
		
		y_pred = []
		for clf in self.model:
			y_pred.append(clf.predict(x))

		self.y_pred = np.array(y_pred)
		plans_index = self.y_pred.sum(axis=0)
		self.predict_result = plans_index
		return self.predict_result
	
	def get_cost(self, y=None):
		if y is None:
			y = self.y_test
			
		if self.predict_result.shape[0] == 0:
			self.predict()
			
		if y.shape[0] == self.predict_result.shape[0]:
			opp_cost  = self.plans_selected- self.plans_selected.min(axis=0)
			self.cost = np.array([opp_cost[i] for i in zip(self.predict_result, y)])
		else:
			print('predict_result与y-test长度不相同')
		
		return self.cost
			
def rock_assessment(target='data2'):
	data = pd.read_excel(r'D:\WPS云文件\代价敏感预测煤岩破坏\数据\现场应用\冲击鉴定.xlsx')
	X = data.loc[:,['σθ / Mpa', 'σc / Mpa', 'σt / MPa', 'σθ/σc', 'σc/σt','Wet']]
	y = data[target]
	
	return X, y


def rock_burst_data(target='data1'):
	"""
    读取冲击地压数据，生成样本X, y

    :param target: target 原始标签，data1：无为0，弱中强为1； data2：无弱为0，中强为1；data3 无弱中为0，强为1
    :return:
    """
	data = pd.read_excel(r'D:\WPS云文件\代价敏感预测煤岩破坏\数据\现场应用\冲击预警-冯.xlsx')
	X = data.iloc[:, 0:6]
	y = data[target]
	return X, y

	
def read_sample(paths, feat_names):
	"""
	读取文件的路径中的声发射特征数据
	:param
	"""
	# folder = Path(r'D:\WPS云文件\代价敏感预测煤岩破坏\数据\100ms')
	# paths = [i for i in folder.iterdir()]
	all_data = pd.DataFrame()
	
	for path in paths:
		print('read data form {}'.format(path.name))
		# path = paths[0]
		data = pd.read_excel(path)
		all_data = all_data.append(data)
	all_data.reset_index(inplace=True)
	x = all_data[feat_names]
	y = all_data['label']

	y = y.map({-1:1, 1:0})
	
	return x, y


	
def experiment_data(file_num=None):
	#
	
	AE_folder = Path(r'D:\WPS云文件\代价敏感预测煤岩破坏\数据\100ms')
	AE_paths = [i for i in AE_folder.iterdir()]
	feat_names = ['MFCC-' + str(i + 1) for i in range(6)]
	paths = np.array(AE_paths[0::2])
	if file_num is not None:
		paths = paths[file_num]
	# if len(paths) ==1:
	# 	paths = [paths]
		
	X, y = read_sample(paths, feat_names)
	
	return X, y

def read_load(paths):
	"""
    从Excel文件中读取加载数据
    :param
    """
	load = pd.DataFrame()
	for path in paths:
		data = pd.read_excel(path)
		load = load.append(data)

	return load


def plot_plans(plans, thresh:np.ndarray, ax=None):
	# TODO 在内部直接完成方案的优选，并画出原始的方案以及最优方案的图，不需要传入格式
	if ax is None:
		fig, ax = plt.subplots()
	x = np.linspace(0, 1, 10)
	plans_y = []
	for i, plan in enumerate(plans):
		print(plan)
		y = plan[0] + (plan[1]-plan[0])*x
		l_y =  plan[0] + (plan[1]-plan[0])*thresh[i]
		plans_y.append(l_y)
		# y = plan[0] + (plan[1] ) * x
		ax.plot(x, y,linewidth=0.5, label='plan-'+str(i+1))
		ax.plot([thresh[i], thresh[i]], [0,l_y], '--', linewidth=0.3,color='#778899')
	
	plan = 	plans[-1]
	l_y = plan[0] + (plan[1] - plan[0])
	plans_y.append(l_y)
	plans_x = np.append(thresh, np.array([1]))
	
	ax.plot(plans_x, plans_y, linewidth=2,color='#DC143C')
	
	ax.plot([1, 1], [0, l_y], '--', linewidth=0.2,color='#778899')
	
	
	plt.ylim((0, plans[0][1]*1.01))
	plt.xlim((0, 1.03))
	plt.legend()
	return fig

def init_model(x_train, y_train, x_test, y_test, plans):
	
	dt_args = dict(criterion='gini',
	               max_depth=5,
	               min_samples_split=0.01,
	               min_samples_leaf=10,
	               )
	csml = CostSensitiveMachineLearning(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, plans=plans,
	                                    base_classifier_para=dt_args)
	
	return csml

def k_fold():
	PLANs = np.array([[0, 100], [2, 60], [40, 40], [20, 30], [8, 40]])
	X, y = experiment_data()
	
	kf = KFold(n_splits=5, shuffle=True, random_state=100)
	# kf = KFold(n_splits=3)
	test_costs = []
	models = []
	
	for train_index, test_index in kf.split(X):
		x_train, y_train = X.loc[train_index,:], y.loc[train_index]
		x_test, y_test = X.loc[test_index,:], y.loc[test_index]
		# train_test_index = {'train_index': train_index, 'test_index': test_index}
		
		models.append(init_model(x_train, y_train, x_test, y_test, PLANs))
	for model in models:
		test_costs.append(model.get_cost().sum())
	
	train_costs = []
	models = []
	for train_index, test_index in kf.split(X):
		x_train, y_train = X.loc[train_index,:], y.loc[train_index]
		x_test, y_test = X.loc[train_index, :], y.loc[train_index]
		models.append(init_model(x_train, y_train, x_test, y_test, PLANs))
	for model in models:
		train_costs.append(model.get_cost().sum())
	
	col_name = 	['kf-'+str(i) for i in range(len(test_costs))]
	cost = pd.DataFrame([train_costs, test_costs], columns=col_name, index=['train', 'test']).T
	cost.to_excel(r'D:\WPS云文件\代价敏感预测煤岩破坏\数据\实验随机5折交叉验证.xlsx')
	
	print(train_costs)

def leave_one():
	#
	PLANs = np.array([[0, 100], [2, 60], [40, 40], [20, 30], [8, 40]])
	AE_folder = Path(r'D:\WPS云文件\代价敏感预测煤岩破坏\数据\100ms')
	save_folder = Path(r'D:\WPS云文件\代价敏感预测煤岩破坏\数据')
	AE_paths = np.array([i for i in AE_folder.iterdir()])
	feat_names = ['MFCC-' + str(i + 1) for i in range(6)]
	
	test_path = AE_paths[1]
	x_test, y_test = read_sample([test_path],feat_names)
	
	train_path = AE_paths[[6, 2, 4]]
	x_train, y_train = read_sample(train_path, feat_names)
	csml = init_model(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, plans=PLANs)
	csml.predict()
	
	fig = model_fig(test_path, csml)
	
	fig.axes[0].set_title(test_path.name)
	

def avrage_ten(PLANs, i, save=None):
	# PLANs = PLANs3
	# i=3
	AE_folder = Path(r'D:\WPS云文件\代价敏感预测煤岩破坏\数据\100ms')
	save_folder = Path(r'D:\wps云文件\代价敏感预测煤岩破坏\数据\7-3不同方案对比')
	AE_paths = np.array([i for i in AE_folder.iterdir()])
	feat_names = ['MFCC-' + str(i + 1) for i in range(6)]
	
	test_path = AE_paths[4]
	x_test, y_test = read_sample([test_path], feat_names)
	x_test = x_test.rolling(10).mean().dropna()
	y_test =  y_test[x_test.index]
	
	train_path = AE_paths[[6, 2, 1]]
	x_train, y_train = read_sample(train_path, feat_names)
	x_train = x_train.rolling(10).mean().dropna()
	y_train = y_train[x_train.index]

	csml = init_model(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, plans=PLANs)
	csml.predict()
	fig = model_fig(test_path, csml)
	# fig.savefig(save_folder / '3-1.jpg', dpi=600)
	plan_fig = plot_plans(csml.plans_selected, csml.plans_thresholds)
	plan_fig.set_size_inches(4.5, 2)
	if save is not None:
		save_name1 = '3-1result_PLAN' + str(i) + '.jpg'
		save_name2 = '方案_PLAN' + str(i) + '.jpg'
		fig.savefig(save_folder/save_name1, dpi=600)
		plan_fig.savefig(save_folder/save_name2, dpi=600)
	plt.close(fig)
	plt.close(plan_fig)

def diff_plans():
	PLANs1 = np.array([[0, 100], [2, 60], [40, 40], [20, 30], [8, 40]])
	PLANs2 = np.array([[0, 100], [2, 60], [40, 40], [28, 38], [10, 40]])
	PLANs3 = np.array([[0, 100], [2, 60], [40, 40], [28, 38], [8, 40]])
	avrage_ten(PLANs1, 1, save='d')
	avrage_ten(PLANs2, 2, save='d')
	avrage_ten(PLANs3, 3, save='d')


def figure(load=None, mode='major'):
	
	fig_size = np.array([12.5, 6.5])
	rect1 = 0.1, 0.15, 0.78, 0.8  # [12.5, 6.5]
	if mode == 'sub':
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


def model_fig(test_path, csml):
	# 绘制模型预测结果与应力对应图
	def inner_fig(test_path, y, mode='major'):
		"""
        :param
        """
		y = pd.Series(y)
		load_folder = Path(r'D:\WPS云文件\代价敏感预测煤岩破坏\数据\应力数据')
		time = y.index / 10
		color = y.map({0: '#0000FF', 1: '#FF8C00', 2: '#FF00FF', 3: '#FF0000'})
		
		load_path = [load_folder / test_path.name]
		load = pd.DataFrame()
		for path in load_path:
			data = pd.read_excel(path)
			load = load.append(data)
	
		fig = figure(load, mode=mode)
		ax_feat = fig.axes[1]
		ax_feat.scatter(time, y, s=0.5, color=color)
		
		ax_feat.yaxis.set_ticks([0, 1, 2, 3])
		ax_feat.yaxis.set_ticklabels(['方案1', '方案2', '方案3', '方案4'])
		
		ax_feat.set_ylim(-3, 4.5)
		
		return fig
	
	save_folder = Path(r'D:\WPS云文件\代价敏感预测煤岩破坏\数据\决策树图')
	AE_folder = Path(r'D:\WPS云文件\代价敏感预测煤岩破坏\数据\100ms')
	fig_name = '11-1.jpg'
	y_pred = csml.predict()
	fig_model = inner_fig(test_path, y_pred)
	return fig_model

	
def mian():
	# X, y = rock_burst_data('data1')
	# X, y = rock_assessment('data1')
	PLANs = np.array([[0, 100], [2, 60], [40, 40], [20, 30], [8, 40]])
	X,y = experiment_data()
	X.reset_index(inplace=True)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	csml = init_model(x_train=X_train, x_test=X_test, y_train=y_train, y_test=y_test, plans=PLANs)
	csml.fit()
	csml.predict()
	cost = csml.get_cost()
	plot_plans(csml.plans_selected, csml.plans_thresholds)


def plot_wave(hello):
	print(hello)
	
# mian()