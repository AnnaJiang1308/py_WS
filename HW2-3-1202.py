# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 21:14:42 2020

@author: 10265
"""


import numpy as np
import pandas as pd
from scipy.io import loadmat
from cvxopt import matrix, solvers
import statsmodels.api as sm
from scipy import stats




#1
#读文件 RIndexFund
R=pd.read_csv('RIndexFund.csv')
#得到三个股票的收益率序列，array格式
R = R.values
#自定义三个股票的权重
w=np.array([0.2,0.3,0.5]).reshape(3,1)
#模拟生成一个指数
IR=np.dot(R,w)

#输入lamda 
lamda=0.5
#输入阿尔法贝塔
alpha=np.array([0.2,0.2,0.2]).reshape(3,1)
beta=np.array([0.5,0.5,0.5]).reshape(3,1)

def indexFund(R,IR,lamda,alpha,beta):
    t=np.shape(R)[0]
    N=np.shape(R)[1]
    P=2*lamda*matrix(np.dot(R.T,R))
    Q1=-2*lamda*matrix(np.dot(R.T,IR))
    c=np.ones((t,1))
    Q2=(1-lamda)*matrix(np.dot(R.T,c))
    Q1=(1-lamda)*matrix(np.dot(R.T,np.ones((t,1))))
    Q=-2*lamda*matrix(np.dot(R.T,IR))-(1-lamda)*matrix(np.dot(R.T,np.ones((t,1))))
    k=-np.identity(N)
    l=np.identity(N)
    G=matrix(np.vstack((k,l)))
    h = matrix(np.vstack((-1 * alpha, beta)))
    A = matrix(np.ones((1,N)))
    b = matrix(1.0)
    sol=solvers.qp(P, Q, G, h, A, b)
    weight = list(sol['x'])
    return weight

weight=indexFund(R,IR,lamda,alpha,beta)
print(weight)
    




#2
#生成以日期和股票代码为索引的closeprice的dataframe
data=loadmat('ClosePrice.mat')
data1=data["ClosePrice"]
stk=data["Stk"][:,0]
date=data["Date"][:,0]

closeprice= pd.DataFrame(data1)
closeprice.columns=stk
closeprice.index=date

#生成以日期和股票代码为索引的hat的dataframe
data=loadmat('hat.mat')
data1=data['FctMat']
stk=data["Stk"][:,0]
date=data["Date"][:,0]

hat= pd.DataFrame(data1)
hat.columns=stk
hat.index=date


out=pd.DataFrame(columns=closeprice.columns,index=closeprice.index)



    
closeprice.fillna(0)

row=0
Flag=0



for i in closeprice.index:
    col=0
    list1=closeprice.loc[i]

    for j in closeprice.columns:
    
        cp=float(closeprice.loc[i,j])
        m=list1[col]

        while Flag==1:
            if int(hat.loc[i,j])==1:
                if float(closeprice.loc[i,j])*1.05<=float(closeprice.iloc[row+1,col]):
                    out.loc[i,j]=1
                elif float(closeprice.loc[i,j])*0.95>=float(closeprice.iloc[row+1,col]):
                    out.loc[i,j]=-1
                else:
                    out.loc[i,j]=0


            elif int(hat.loc[i,j])==0:
                if (float(closeprice.loc[i,j])*1.1)<= float(closeprice.iloc[row+1,col]):
                    out.loc[i,j]=1
                elif float(closeprice.loc[i,j])*0.9>= float(closeprice.iloc[row+1,col]):
                    out.loc[i,j]=-1
                else:
                    out.loc[i,j]=0
        col++

    if Flag==0:
        Flag=1
    row++


#3

#设定函数：将hat矩阵与任一时间/股票的closeprice_data矩阵进行匹配,并输出涨跌停矩阵
#参见第一题
def bound(closeprice,hat):
    #输入1：以日期和股票代码为索引的closeprice的dataframe
    #输入2：以日期和股票代码为索引的hat的dataframe
    #输出：以日期和股票代码为索引的涨跌停板的dataframe
    out=pd.DataFrame(columns=closeprice.columns,index=closeprice.index)
    row=0
    for i in closeprice.index:
        col=0
        for j in closeprice.columns:         
            if int(hat.loc[i,j])==1:
                if float(closeprice.loc[i,j])*1.05<=float(closeprice.iloc[row+1,col]):out.loc[i,j]=1
                elif float(closeprice.loc[i,j])*0.95>=float(closeprice.iloc[row+1,col]):out.loc[i,j]=-1
                else:out.loc[i,j]=0
            elif int(hat.loc[i,j])==0:
                if float(closeprice.loc[i,j])*1.1<=float(closeprice.iloc[row+1,col]):out.loc[i,j]=1
                elif float(closeprice.loc[i,j])*0.9>=float(closeprice.iloc[row+1,col]):out.loc[i,j]=-1
                else:out.loc[i,j]=0
            col+=1
        row+=1
    return out


#生成以日期和股票代码为索引的closeprice的dataframe
data=pd.read_csv("C:\\Users\\susan\\Desktop\\多因子模型作业及数据2020\\ClosePriceAdj.csv",sep=',',index_col=0)

#生成以日期和股票代码为索引的hat的dataframe
data=loadmat("C:\\Users\\susan\\Desktop\\多因子模型作业及数据2020\\hat.mat")
data1=data['FctMat']
stk=data["Stk"][:,0]
date=data["Date"][:,0]

hat= pd.DataFrame(data1)
hat.columns=stk
hat.index=date

#将hat矩阵与data矩阵进行匹配,并输出涨跌停矩阵
#去除索引，后续处理过程中保证涨跌停矩阵与data矩阵同步变动
bound=bound(data,hat).values
data = data.values


def daySingle(data,J,K,M): # 一次JK策略函数
#输入1： 价格数据:data(J+K+2 *N)  数据类型为array
#输入2：排序期 J  1*1
#输入3：持有期 K  (1*1)
#输入4：赢家组合和输家组合包含的股票数 M 1*1
#输出：输家与赢家平均收益:rLoserAnnual(1*1),rWinnerAnnual(1*1)
#step1:计算排序期每个股票的收益率
# ret为一维数组
    ret = data[J] / data[0] - 1
    #data[J]表示原始价格矩阵的第J+1行，data[0]表示原始价格矩阵的第一行
#step2:对ret排序，得到收益率从小到大的股票序号
    idx = np.argsort(ret) 
#step3:按照序号排序，得到重新排序后的价格矩阵，矩阵左侧为输家，右侧为赢家;同时对bound矩阵进行重新排序
    priceOrder = data[:, idx]
    bound=bound[:, idx]
#step4:按照排序后的矩阵计算持有期每个股票的收益率   
    rHold = priceOrder[J+K] / priceOrder[J] - 1
    #bound[J,i]为购买日第i个股票的涨跌停情况，为1or-1就从array中删除，再取左右的M个股票作为赢家/输家组合
    for i in range(rHold.shape[0]): #共有rHold.shape[0]个股票
        if bound[J,i]==1 or bound[J,i]==-1:rHold=np.delete(rHold,i,axis=0)
#step5:计算持有期内输家组合的收益率和赢家组合的收益率   
    rLoser = rHold[:M].mean()
    rWinner = rHold[-M:].mean()
#step6:计算年化的输家组合收益率、赢家组合的收益率 
    rLoserAnnual=(1+rLoser)**(252/K)-1
    rWinnerAnnual=(1+rWinner)**(252/K)-1
    return rLoserAnnual,rWinnerAnnual

def dayMean(data,J,K,M): # 多次JK策略均值
#输入：价格数据:data(T*N) T>>J+K+1， 策略参数:J K M (1*1)
#输出：输家与赢家平均收益，rLoserAnnual(1*1),rWinnerAnnual(1*1)
    T,N = data.shape
    loser,winner = np.zeros(T-J-K),np.zeros(T-J-K)
    for i in range(T-J-K): # 循环T-J-K次，分别计算T-J-K个赢家与输家收益
        data1=pd.DataFrame(data[i:i+1+J+K,:])
        data1.dropna(axis=1, how='any')#删去该时刻存在空白值
        data1=data1.values
        #删去j+2时刻涨停的股票
        res =  daySingle(data1, J, K ,M)
        loser[i] = res[0]
        winner[i] = res[1]
    return loser.mean(), winner.mean()

def dayTab(data,M): # JK策略表格，J&K均取之于1-10
#输入1：价格数据:data(T*N) T>>J+K+1 
#输入2 M 赢家和输家组合含有的股票个数 (1*1)
#输出：JK策略赢家组合收益率表格（10*10），输家组合收益率表格（10*10），赢家减输家收益率表格
    rLoserAnnualMeanMatrix=np.zeros((10,10))
    rWinnerAnnualMeanMatrix=np.zeros((10,10))
    for J in range(10):
        for K in range(10):
            rLoserAnnualMeanMatrix[J,K],rWinnerAnnualMeanMatrix[J,K]=dayMean(data,J+1,K+1,M) 
            # 定位参数为J，K（0~J-1，0~K-1）,函数参数为J+1，K+1（1~J，1~K）
    rWMLAnnualMeanMatrix=rWinnerAnnualMeanMatrix-rLoserAnnualMeanMatrix
    return  rLoserAnnualMeanMatrix, rWinnerAnnualMeanMatrix, rWMLAnnualMeanMatrix
#调用momTab,momTab调用momMean，momMean调用momSingle
M = 10
dayTab=dayTab(data,M)


#4

data = pd.read_csv("C:\\Users\\susan\\Desktop\\多因子模型作业及数据2020\\capm_Test.csv")

def regress(data, para1, para2):
    Y = data[para1]
    X = data[para2]
    X['intercept'] = 1.
    result = sm.OLS(Y, X).fit()
    return result.params

beta = data.groupby('Stkcd').apply(regress, 'ret', ['mkt'])
data2 = beta.sort_values(by='mkt')
stk_rank = data2._stat_axis.values.tolist()
data2['Stkcd']=stk_rank
data2 = data2.reset_index(drop=True)

data3 = pd.DataFrame(data[data['Stkcd']==1].loc[:,['year','month','ym','mkt']])
data3.set_index(["ym"], inplace=True)

temp = pd.DataFrame(data)
temp.set_index(['ym'],inplace=True)

for i in range(len(stk_rank)):
    data3['Stock'+str(stk_rank[i])] = temp[temp['Stkcd']==stk_rank[i]].loc[:,'ret']

portfolio1 = data3.iloc[:,3:]
portfolio_L = portfolio1.iloc[:,0:10]
portfolio_H = portfolio1.iloc[:,-10:]
print(portfolio_H)
print(portfolio_L)

L1 = portfolio_L.mean(axis=1)
H1 = portfolio_H.mean(axis=1)
print("L1",L1)
print("H1",H1)

L_mean = np.mean(L1)
L_var = np.var(L1)
H_mean = np.mean(H1)
H_var = np.var(H1)
print("L_mean:",L_mean,"\L_var:",L_var,"\nH_mean:",H_mean,"\nH_var:",H_var)

result = stats.ttest_ind(L1,H1)
print("T test:",result)