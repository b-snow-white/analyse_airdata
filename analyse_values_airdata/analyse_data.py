# -*- coding: utf-8 -*-
"""
@time:2018/7/26 17:55

@author: BX
"""
#过滤掉不符合规则的数据
import pandas as pd
import os
abspath=os.getcwd()
abspath=os.path.join(abspath,r'data')
datafile=os.path.join(abspath,'air_data.csv')
cleanedfile=os.path.join(abspath,'data_cleaned.xls')
data=pd.read_csv(datafile,encoding='utf-8')
data=data[data['SUM_YR_1'].notnull()*data['SUM_YR_2'].notnull()]#票价非空保留
#只保留票价非零的，或者平均折扣率为0与总飞行公里数为0的记录
index1=data['SUM_YR_1']!=0
index2=data['SUM_YR_2']!=0
index3=(data['SEG_KM_SUM']==0)&(data['avg_discount']==0)
data=data[index1|index2|index3]
#print(data.columns)

#对属性进行选择，删掉不相关或者弱相关的属性，例如：性别，会员卡号
data1=data[['LOAD_TIME','FFP_DATE','LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM','avg_discount']]
data1.to_excel(cleanedfile)#导出结果


#对属性进行变换提取，整理数据,使符合LRFMC五个指标模型
import time
#print(type(data1['LOAD_TIME'][0]))
'mktime()返回的是秒数，后期可进行计算'
data1['FFP_DATE'] = [time.mktime(time.strptime(i,"%Y/%m/%d")) for i in data1['FFP_DATE']]
data1['LOAD_TIME'] = [time.mktime(time.strptime(i,"%Y/%m/%d")) for i in data1['LOAD_TIME']]
data1['L']=round((data1['LOAD_TIME']-data1['FFP_DATE'])/(30*24*60*60),2)
#会员入会时间距离观测时间的月数
data1['R']=round((data1['LAST_TO_END'])/30,2)#最近一次消费时间间隔
data1['F']=data1['FLIGHT_COUNT'].astype('int')#客户在观测窗口内乘坐飞机的次数
data1['M']=data1['SEG_KM_SUM'].astype('int')#观测窗口的总飞行公里数
data1['C']=round(data1['avg_discount'],2)#平均折扣率
data_result=data1[['L','R','F','M','C']]
createdatafile=os.path.join(abspath,r'data_create.xls')
data_result.to_excel(createdatafile)
#print(data_result)
#对数据进行描述
#print(data_result.describe(percentiles=[],include='all').T)

#数据标准化（标准差）
data_to_zs=pd.DataFrame(data_result)
data=(data_to_zs-data_to_zs.mean(axis=0))/(data_to_zs.std(axis=0))#进行数据标准化
data.columns=['Z_'+i for i in data_to_zs.columns]#表头重命名
#print(data)
zs_file=os.path.join(abspath,r'data_zs.xls')
data.to_excel(zs_file)

#模型构建
from sklearn.cluster import KMeans#导入K均值聚类算法
inputfile=zs_file#待聚类的数据文件
k=5  #需要结合实际情况进行确定
#读取文件并进行聚类
data=pd.read_excel(inputfile)
#调用k均值算法，进行聚类分析
#interation=500#最大迭代次数
kmodel=KMeans(n_clusters=k,n_jobs=1)#n_jobs是并行数，一般等于CPU个数较好
kmodel.fit(data)#训练模型
#print(kmodel.cluster_centers_)#查看聚类中心
#print(kmodel.labels_)#查看个样本对应的类型
r1=pd.DataFrame(kmodel.cluster_centers_)
r2=pd.Series(kmodel.labels_).value_counts()#查看每一类对应的个数
r=pd.concat([r2,r1],axis=1)
index=['客户群1','客户群2','客户群3','客户群4','客户群5']
columns=['聚类个数','ZL','ZR','ZF','ZM','ZC']
r.index=index
r.columns=columns
#客户聚类结果
clusterfile=os.path.join(abspath,r'cluster.xls')
r.to_excel(clusterfile)
#print(type(r))
#print(r)连
#详细输出原始数据和聚类类别
result=pd.concat([data_result,pd.Series(kmodel.labels_,index=data_result.index)],axis=1)
result.columns=list(data_result.columns)+['聚类类别']
clusterdatafile=os.path.join(abspath,r'cluster_data.xls')
result.to_excel(clusterdatafile)


#分群的概率密度图
import matplotlib.pyplot as plt
def dednsity_plot1(data):#定义作图函数
    plt.figure(figsize=(30,30))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用了来正常显示负号
    p=data.plot(kind='kde',linewidth=2,subplots=True,sharex=False)
    [p[i].set_ylabel('密度') for i in range(k)]
    plt.legend()
    return plt
#pic_output=os.path.join(os.path.join(os.getcwd(),'plot_result'),r'pd_')#概率密度图文件名前缀
for i in range(k):
    plot_density=dednsity_plot1(data_result[result['聚类类别']==i])
    plot_density.show()

#画雷达图进行分析
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False
colors = ['b', 'r', 'g', 'm', 'y']
inputfile=r'H:\data_test\chapter7\demo\tmp\out_cluster.xls'
r=pd.read_excel(inputfile)
print(r)
labels=r.columns[1:]#类别标签
kinds=r.index#种类
#雷达图要保证数据闭合,所以再添加第一列
result=pd.concat([r.iloc[:,1:],r.iloc[:,1]],axis=1)
centers=np.array(result)
n=len(labels)
#分割圆周长，并且让其闭合
angle=np.linspace(0,2*np.pi,n,endpoint=False)
print(angle)
angle=np.concatenate((angle,[angle[0]]))#闭合
#画图
fig=plt.figure()#画布
ax=fig.add_subplot(111,polar=True)#以极坐标的形式绘制图形

#画线
for i in range(len(kinds)):
    ax.plot(angle,centers[i],linewidth=2,label=kinds[i],color=colors[i])
    ax.fill(angle,centers[i],facecolor=colors[i],alpha=0.25)#alpha为透明度
#添加属性标签
ax.set_thetagrids(angle*180/np.pi,labels)
box = ax.get_position()
#如果你想将图例放上面就box.height*0.8，放右边就box.width*0.8
ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])
#显示图例的位置
ax.legend(loc='center left', bbox_to_anchor=(0.9, 0.1))
ax.set_title("客户群特征分析图", va='bottom', fontproperties="SimHei")
pic_output_radar=os.path.join(os.path.join(os.getcwd(),'plot_result'),r'radar.jpg')#雷达图文件名前缀
plt.savefig('%s'%pic_output_radar)
plt.show()
#分析结束
'''
本案例将客户分成五个等级：重要保持客户，重要发展客户，重要挽留客户，一般客户，低价值客户
'''


