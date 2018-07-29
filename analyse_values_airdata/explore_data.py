# -*- coding: utf-8 -*-
"""
@time:2018/7/26 11:08

@author: BX
"""

#数据探索分析
import pandas as pd
import os
#转换为绝对路径进行操作
abspath=os.getcwd()
abspath=os.path.join(abspath,r'data')

datafile=os.path.join(abspath,r'air_data.csv')
resultfile=os.path.join(abspath,r'explore_result.csv')#数据探索结果表
data=pd.read_csv(datafile,encoding='utf-8')
explore=data.describe(percentiles=[],include='all').T
print(explore)
#percentiles指定计算多少的分位数表（如1/4、中位数等）;T转置
explore['null']=len(data)-explore['count']#describe计算非空数值，需要手动计算空数值
print(len(data))
explore=explore[['null','max','min']]
explore.columns=['空数值数','最大值','最小值']#表头重命名
'''
describe()函数自动计算的字段有count(非空数值)，unique(唯一数值)
top（频数最高者），freq(最高频数),mean(平均值),std(方差),min(最小值)
50%(中位数),max(最大值)
'''
explore.to_csv(resultfile)#导出结果

