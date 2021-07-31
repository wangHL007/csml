# !/usr/bin/python3
# -*-coding:utf-8-*-
# Author: 王洪磊
# Email: wang_hl007@163.com
# CreatDate: 2021/7/29 10:25
# Description:

import re
import pandas as pd
from pathlib2 import Path


def get_raw_ae_paths(folder):
    folder = Path(folder)
    if folder.is_dir():
        file_paths = list(folder.glob('*[0-9]A.dat'))
      
        file_times = list(map(lambda x: int(re.search('D[0-9]{6}', x.name).group()[1:]), file_paths))
    
        paths = pd.DataFrame.from_records([file_times, file_paths], index=['time', 'path']).sort_values(by='time', axis=1)
        path_text = [str(i) for i in paths.loc['path', :].values.tolist()]
        return path_text
    else:
        print('{} 文件夹不存在')


