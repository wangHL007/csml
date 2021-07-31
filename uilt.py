# !/usr/bin/python3
# -*-coding:utf-8-*-
# Author: 王洪磊
# Email: wang_hl007@163.com
# CreatDate: 2021/7/26 16:29
# Description:

from PyQt5.QtWidgets import QDialog, QLabel, QProgressBar
from PyQt5.QtCore import Qt

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

def p(f, v):
	print(f, 'ss', v)
