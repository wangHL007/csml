# !/usr/bin/python3
# -*-coding:utf-8-*-
# Author: 王洪磊
# Email: wang_hl007@163.com
# CreatDate: 2021/7/26 16:29
# Description:

from PyQt5.QtWidgets import QDialog, QLabel, QProgressBar
from PyQt5.QtCore import Qt

import xml.etree.cElementTree as et
import pandas as pd


class WarringDialog(QDialog):

	def __init__(self, info, parent=None):
		super(WarringDialog, self).__init__(parent=parent)

		lb = QLabel(info, self)
		lb.move(10, 20)
		self.setWindowTitle("缺失数据集")
		self.setWindowModality(Qt.ApplicationModal)
		self.exec_()


def progress_bar(minimum, maximum, parent, size=None, move=None):
	if size is None:
		size = (300, 30)
	if move is None:
		move = (130, 20)
	pgb = QProgressBar(parent)
	pgb.move(*move)
	pgb.resize(*size)
	pgb.setMinimum(minimum)
	pgb.setMaximum(maximum)

	return pgb


def read_xml(file_name):
	# 读取xml文件，放到dataframe df_xml中
	xml_tree = et.ElementTree(file=file_name)  # 文件路径
	dfcols = ['sentence', 'opinionated', 'polarity']
	df_xml = pd.DataFrame(columns=dfcols)
	root = xml_tree.getroot()

	for sub_node in root:
		for node in sub_node:
			print(node, node.tag, node.attrib, node.text)
			sentence = node.text
			print(sentence)
			# opinionated = node.attrib.get('component')
			# polarity = node.attrib.get('mapping')
			# print([sentence, opinionated, polarity])

			# df_xml = df_xml.append(
			# 	pd.Series([sentence, opinionated, polarity], index=dfcols),
			# 	ignore_index=True)
