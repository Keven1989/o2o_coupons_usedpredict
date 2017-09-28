# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 23:17:46 2017

@author: yuankun
"""

from pandas import Series,DataFrame
import pandas as pd

############################Series一维数组############################
obj = Series([4,7,-5,3])
print(obj)

obj.values # 获取数组
obj.index #获取索引

obj2 = Series([4,7,-5,3],index = ['a','b','c','d']) #创建索引
print(obj2)
obj2.index #获取索引

obj2['a'] #通过索引获得数据
obj2['d'] = 2 #通过索引获得数据，并改变值
obj2[['c','a','d']]

obj2 #numpy数值运算都会保留索引和值之间的链接
obj2[obj2 > 0] #获取大于0的值
obj2*2 #在原来的数值上乘以2
import numpy as np 
np.exp(obj2) #以自然常数e为底的指数函数

'b' in obj2 #Series可以看成是一个定长的有序字典，因为它是索引值到数据值的一个映射。
'e' in obj2

sdata = {'Ohio':35000,'Texas':71000,'Oregon':16000,'Utah':5000} #注意这里冒号不要写成等号了
obj3 = Series(sdata) #也可以将字典创建Series
obj3
sdatas = ['xxyyy','Ohio','Texas','Oregon']
obj4 = Series(sdata, index = sdatas)
obj4 #定义索引来匹配值，xxyyy找不到值为NaN

pd.isnull(obj4) #检查缺失值
pd.isnull(obj4['xxyyy'])
pd.notnull(obj4) #检查非缺失值

obj4.isnull
obj4.notnull

obj3
obj4
obj3 + obj4 #Series一个重要功能是在运算过程中会自动对齐不同索引的数据

obj4.name = 'population'
obj4.index.name = 'state' #给数据对象及其索引命名
obj4
obj4.index
obj4.values

obj
obj.index = ['bob','steve','jeff','ryan']
obj #Series的索引可以通过赋值的方式来改变


############################DataFrame表格型数据结构############################
#DataFrame既有行索引也有列索引，他可以看成是由Series组成的字典
from pandas import DataFrame,Series
import numpy as np
data = {
        'state':['bob','steve','jeff','ryan'],
        'year':[2000,2001,2001,2002],
        'pop':[1.5,1.7,3.6,2.4]
        } #创建字典
data
frame = DataFrame(data)
frame

DataFrame(data,columns = ['state','year','pop']) #按照指定的列序列显示数据

frame2 = DataFrame(data,
          columns = ['state','year','pop','ta'],
          index = ['a','b','c','d']
          ) #指定的列在数据找不到就会显示NaN
frame2           
frame2.columns

frame2['state'] #获取列的一个Series

frame2 
frame2.ix['c'] #获取行的一个Series

frame2['ta'] = 15 #给列赋值
frame2
frame2['ta'] = ([0,1,2,3]) #修改列的值
frame2['ta'] = (0,1,2,3) #修改列的值
frame2['ta'] = np.arange(4.)
frame2

val = Series([-1.2,-1.5,-1.7],index = ['a','c','d'])
frame2['ta'] = val
frame2 #没匹配上的是NaN

val = Series([-1.2,-1.5,-1.7,1.5,1.0],index = ['a','e','d','b','c']) 
frame2['ta'] = val #给列赋值
frame2 

frame2['eastern'] = frame2.state == 'bob'
frame2

del frame2['eastern'] #del用于删除列
frame2.columns

pop = {'Nevada':{2001:2.4,2002:2.9},
       'Ohio':{2000:1.5,2001:1.7,2002:3.6}} #嵌套字典，字典中的字典
frame3 = DataFrame(pop) #将嵌套字典转换成DataFrame，外层健是列，内层健是行索引
frame3.T #行列转置
DataFrame(pop,index = [2001,2002,2003]) #显示指定索引

frame3[:]
frame3[:-1]
frame3['Ohio']
frame3['Ohio'][:-1]
pdata = {'Ohio':frame3['Ohio'][:-1],
         'Nevada':frame3['Nevada'][:2]}
DataFrame(pdata)

frame3.index.name = 'year';frame3.columns.name = 'state'
frame3 #给DataFrame的index和columns的name属性

frame3.values #跟Series一样，values属性也会以二维ndarry的形式返回DataFrame中的数据
#Numpy库中的矩阵模块为ndarray对象

frame2
frame2.values #如果DataFrame各列数据类型不同，则值数组的数据类型就会选用能兼容所有列的数据类型



####################第三部分 索引对象##########################################
import numpy as np
import pandas as pd
obj = Series(range(3), index = ['a','b','c'])
obj
obj.index
index = obj.index
index[1:]
index[1] = 'd' #索引不能被修改，该语句执行会报错。索引不能修改非常重要，这才能使index对象在多个数据结构之间可以共享

index = pd.Index(np.arange(3)) #Pandas主要的Index对象,注意这里I是大写的
obj2 = Series([1.5,-2.5,0],index = index)
obj2.index is index


frame3
'Ohio' in frame3.columns
2003 in frame3.index



####################第四部分 基本功能##########################################
from pandas import Series,DataFrame

obj = Series([4.5,7.2,-5.3,3.6],index = ['d','b','a','c'])
obj2 = obj.reindex(['a','b','c','d','e']) #reindex将会根据新索引进行重排
obj.reindex(['a','b','c','d','e'],fill_value = 0)

obj3 = Series(['bule','purple','yellow'],index = [0,2,4])
obj3.reindex(range(6),method='ffill') #按前向填充
obj3.reindex(range(6),fill_value = 0) #例如时间序列的有序数据，会用到该方面进行插值处理
obj3.reindex(range(6),method='pad')  #同ffill
obj3.reindex(range(6),method='bfill')  #同ackfill，按后向填充

frame = DataFrame(np.arange(9).reshape((3,3)),index = ['a','c','d'],
                  columns = ['Ohio','Texas','California'])
frame
frame2 = frame.reindex(['a','b','c','d'])
print(frame2)
frame2 = frame.reindex(['a','b','c','d'],method = 'ffill')
print(frame2)

states = ['Texas','Utah','California']
frame.reindex(columns = states)
frame.reindex(index=['a','b','c','d'],method = 'ffill',columns = states)
frame.ix[['a','b','c','d'],states] #与上面reindex重新索引一样


####################第五部分 丢弃指定轴上的项#######################################
obj = Series(np.arange(5.),index = ['a','b','c','d','e'])
new_obj = obj.drop('c') #删除Series某行
new_obj

obj.drop(['a','b'])

data = DataFrame(np.arange(16).reshape((4,4)),
                 index = ['Texas','Utah','California','Ohio'],
                 columns = ['One','Two','Three','Four'])
data.drop(['Texas','Utah']) 
data.drop(['Texas','Utah'],axis=0)   
data.drop(['One','Two'],axis=1)               

####################第六部分 索引选取过滤#######################################
obj = Series(np.arange(4.),index = ['a','b','c','d'])
print(obj)
obj['b']
obj[1]
obj[2:3]
obj[['b','a','d']]
obj[[1,3]]
obj[obj < 2]

obj[1:3]
obj['b':'d']
obj['b':'d'] = 5
print(obj)


data = DataFrame(np.arange(16).reshape((4,4)),
                 index = ['Texas','Utah','California','Ohio'],
                 columns = ['One','Two','Three','Four'])
print(data)

data['Two'] #选取two列
data['Utah'] #不能选取行，错误语句 修正为data[1:2]

data.ix['Two'] #错误语句，不能选取列  修正为data.ix[:,'Two']
data.ix['Utah'] #选取Utah行


data['Two']
data[['Two','One']]
data[:2]
data[data['Three'] > 10]

data <5
data[data<5]
data[data<5] = 0
print(data)

data.ix['California',['Three','Four']] #行和列都可以用ix索引,前面是行,后面是列
data.ix[1:3,['Three','Four']] 
data.ix[1:3,1:3]
data.ix[['Ohio','Texas'],[3,0,1]] #按指定顺序重新索引

data.ix[2]
data.ix[:'California',:'Three']
data.ix[data.Three >5 ,:4]
data.ix[data['Three'] >5 ,:4]


####################第七部分 算术运算和数据对齐#################################
s1 = Series([7.3,-2.5,3.4,1.5],index=['a','c','d','e'])
s2 = Series([-2.1,-3.6,-1.5,4,3.1],index=['a','c','e','f','g'])
print(s1,s2)
s1+s2 #自动对齐索引


data1 = DataFrame(np.arange(9.).reshape((3,3)),
                 index = ['Texas','California','Ohio'],
                 columns = list('bcd'))
data2 = DataFrame(np.arange(12).reshape((4,3)),
                 index = ['Texas','Utah','California','Ohio'],
                 columns = list('bde'))        

print(data1,data2)
data1 + data2  #自动对齐行和列索引


####################第八部分 在算术方法中填充值#################################
#在对不同索引的对象进行算术运算时，你可能希望当一个对象中某个轴标签在另一个对象中
#找不到时填充一个特殊值（比如0）：
from pandas import DataFrame,Series
import numpy as np


df1 = DataFrame(np.arange(12.).reshape((3,4)),columns = list('abcd'))
df2 = DataFrame(np.arange(20.).reshape((4,5)),columns = list('abcde'))

print(df1)
print(df2)

#将df1和df2相加时，没有重叠的位置将会产生NA值：
df1 + df2
#使用df1的add方法，传入df2以及一个fill_value参数：
df1.add(df2, fill_value = 0)
#与此类似，在对Series或DataFrame重新索引时，也可以指定一个填充值：
df1.reindex(columns = df2.columns ,fill_value = 0)

#灵活的算术方法
add 加法
sub 减法
div 除法
mul 乘法


####################第九部分 Series与DataFrame之间的运算#######################
#先看看numpy中一个二维数组与其某行之间的差：
ar = np.arange(12.)
arr = ar.reshape((3,4))
arr #数组
arr[0]
arr - arr[0] #向下广播

#再看看Series与DataFrame之间的运算，与上类似
frame = DataFrame(np.arange(12.).reshape((4,3)),columns = list('bde'),
                  index = ['U','O','T','X'])
frame
series = frame.ix[0]
series
frame - series 
frame.sub(series,axis=1)

#如果某个索引在DataFrame的列或Series的索引中找不到，则参与运算的两个对象就会被重新
#索引以形成并集：
series2 = Series(range(3),index = ['b','e','f'])
frame + series2

#如果希望匹配行且在列上广播，则会必须使用算术运算方法。例如：
series3 = frame['d']
series3
frame
frame.sub(series3,axis=0) #axis=0表示列，axis=1表示行



####################第十部分 函数应用与映射####################################
#numpy的ufuncs（元素级数组方法）也可用于操作pandas对象：
frame = DataFrame(np.random.randn(4,3),columns = list('bde'),
                  index = ['U','O','T','X'])
frame
np.abs(frame)

#另一个常见的操作是，将函数应用到由各列或行所形成的一维数组上。DataFrame的apply
#方法即可实现此功能：
f = lambda x: x.max() -x.min()
frame.apply(f)
frame.apply(f, axis = 1)

#许多最为常见的数组统计功能都被实现成DataFrame的方法（如SUM和MEAN）,由此无需使用apply方法
def f(x):
    return Series([x.min(),x.max()],index = ['min','max'])

f(frame)  
frame.min()
frame.max()
frame.min(axis=1)
frame.max(axis=1)

#此外元素级的Python函数也可以用。假如想得到frame中各个浮点值的格式化字符串，可以使用
#applymap方法：
format = lambda x: '%.2f' % x
frame.applymap(format)
#之所以叫做applymap,是因为Series有一个用于应用元素级函数的map方法：
frame['e'].map(format)


    










