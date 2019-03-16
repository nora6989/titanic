import pandas as pd

ctx='C:/Users/ezen/WEEKEND_TENSORFLOW/titanic/data/'
train = pd.read_csv(ctx+'train.csv')
test = pd.read_csv(ctx+'test.csv')
train.head() 
test.head() 

train.columns

"""
Index(['PassengerId', : 승객번호
'Survived', : 생존여부(1:생존, 0:사망) 
'Pclass', : 승선권클래스(1~3:1~3등석)
'Name', : 승객 이름
'Sex', : 승객 성별
'Age', : 승객 나이
'SibSp', : 동반한 형제자매, 배우자 수 
'Parch', : 동반한 부모, 자식 수
'Ticket', : 티켓의 고유넘버  
'Fare',: 티켓의 요금 
'Cabin',: 객실 번호 
'Embarked': 승선한 항구명(C:Cherbourg, S:Southampton, Q: Queenstown)
],
      dtype='object')
"""

import seaborn as sns 
import matplotlib.pyplot as plt

# 1. 생존여부와 조건과의 상관관계를 파악한다 

f, ax = plt.subplots(1,2, figsize=(18,8))
train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=train, ax=ax[1])
ax[1].set_title('Survived')
plt.show()

"""
탑승객의 61.6% 사망, 38.4% 생존
"""

# 2. 성별에 따른 생존률 
f, ax = plt.subplots(1,2, figsize=(18,8))
train['Survived'][train['Sex']=='male'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%', ax=ax[0], shadow=True)
train['Survived'][train['Sex']=='female'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%', ax=ax[1], shadow=True)
ax[0].set_title('male')
ax[1].set_title('female')

plt.show()

"""
남성: 생존 18.9%, 사망 81.1% 
여성: 생존 74.2%, 사망 25.8% 
"""

# 3. 승선권클래스에 따른 생존률 
df_1 = [train['Sex'],train['Survived']]
df_2 = train['Pclass']
pd.crosstab(df_1, df_2, margins=True)
# 그래프가 아닌 판다스의 자체 table 기능을 사용한 분석
"""
Pclass             1    2    3  All
Sex    Survived                    
female 0           3    6   72   81
       1          91   70   72  233
male   0          77   91  300  468
       1          45   17   47  109
All              216  184  491  891
"""


# 4. 배를 탄 항구에 따른 생존률 Embarked
f, ax = plt.subplots(2,2, figsize=(20,15))
sns.countplot('Embarked', data=train, ax=ax[0,0])
ax[0,0].set_title('No of Passengers Boarded')
sns.countplot('Embarked', hue='Sex', data=train, ax=ax[0,1])
ax[0,1].set_title('Male-Femail for Embarked')
sns.countplot('Embarked', hue='Survived', data=train, ax=ax[1,0])
ax[1,0].set_title('Embarked vs Survived')
sns.countplot('Embarked', hue='Pclass', data=train, ax=ax[1,1])
ax[1,1].set_title('Embarked Pclass')
plt.show()

""" 
 절반 이상의 승객이 S항구에서 배를 탔으며, 여기에서 탑승한 승객의 70% 가량이 남성임. 
 남성의 사망률이 여성보다 훨씬 높았기에, 자연스럽게 S항구에서 탑승한 승객의 사망률이 높게 나왔음. 
 C항구에서 탑승한 승객들은 1등 객실 승객의 비중 및 생존률이 높은 것으로 보여서 
 이 도시는 부유한 도시라는 것을 예측할 수 있음 
"""

# ************************************
# 결과 도출을 위한 전처리(Pre-processing)
# *************************************

"""
가장 강한 상관관계를 가지고 있는 성별, 객실 등급, 탑승 항구 세가지 정보를 가지고 모델을 구성하고자 함 
"""

# 결측치 제거 
# 비어있는 데이터를 제거하여 연산에서의 오류를 방지함 

train.info 
# [891 rows x 12 columns]

train.isnull().sum()
"""
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177 # 177개의 결측치가 있음. 
                        나이에 따른 생존여부가 상관관계가 있을 듯 하여 임의의 데이터로 채워 넣어서 해결 
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687 # 객실번호에 따른 생존여부가 상관관계가 있을 듯 하여 데이터를 채우려 했으나, 
                         임의의 데이터를 산정하기 어렵고, Pclass를 대체하여 분석가능하므로 제거함 
Embarked       2  # 승선한 항구의 결측치는 예상하기 어려우나, 가장 많이 승선한 S항구를 임의값으로 대체함 
dtype: int64
"""

import matplotlib.pyplot as plt 
import seaborn as sns

def bar_chart(feature): 
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df =pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar', stacked =True, figsize=(10,5))
    plt.show()

bar_chart('Sex')
bar_chart('Pclass')      # 생존자 1등석, 사망자 3등석 
bar_chart('SibSp')      # 동반한 형제자매, 배우자  
bar_chart('Parch')      # 동반한 부모, 자식수 
bar_chart('Embarked')

# Cabin은  null 값이 많아서, Ticket은 전혀 관계가 없어서 삭제 

train = train.drop(['Cabin'], axis =1)  
test =  test.drop(['Cabin'], axis =1)  # 트레이닝에서 지우면 테스트에서도 반드시 지워라 
train = train.drop(['Ticket'], axis =1)  
test =  test.drop(['Ticket'], axis =1)  # 트레이닝에서 지우면 테스트에서도 반드시 지워라 
train.columns
test.columns 

# Embarked에 있는 null 2개 처리 

s_city = train[train['Embarked']=='S'].shape[0]
print("S",s_city) # S 644
s_city = train[train['Embarked']=='Q'].shape[0]
print("Q",s_city) # Q 77
s_city = train[train['Embarked']=='C'].shape[0]
print("C",s_city) # C 168

#  na에 값을 채우기 
train = train.fillna({'Embarked':"S"})  #{":"} 

# 머신러닝에서 모든 값은 숫자로만 인식함
# 따라서 S, C, Q를 숫자로 1,2,3으로 가공함 

city_mapping = {"S":1,"C":2,"Q":3 }
train["Embarked"] = train["Embarked"].map(city_mapping)
test["Embarked"] = test["Embarked"].map(city_mapping) 
train.head()
test.head()


# Name 값 가공하기 
"""
승객이름에서 Mr, Mrs가 있음 
"""

combine = [train, test]
for dataset in combine:
     dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
pd.crosstab(train['Title'], train['Sex'])

"""
Sex       female  male
Title                 
Capt           0     1
Col            0     2
Countess       1     0
Don            0     1
Dr             1     6
Jonkheer       0     1
Lady           1     0
Major          0     2
Master         0    40
Miss         182     0
Mlle           2     0
Mme            1     0
Mr             0   517
Mrs          125     0
Ms             1     0
Rev            0     6
Sir            0     1
"""
#ㄱ+한자키로 ＼찾음. but 인식못함 \로 써야함 
# 정규표현식 '([A-Za-z]+)\.'

# Mr, Mrs, Miss, Royal, Rare, Master 6개로 줄여서 정리 
for dataset in combine:
     dataset['Title'] = dataset['Title'].replace(['Capt','Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer'], 'Rare') 
     dataset['Title'] = dataset['Title'].replace(['Countess','Lady', 'Sir'], 'Royal')  
     dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')  
     dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
     dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title','Survived']].groupby(['Title'], as_index=False).mean()
# groupby는 같은 값을 가지는 인스턴스를 묶어서 연산하라(mean이면 평균 구하기) 
"""
 Title  Survived
0  Master  0.575000
1    Miss  0.702703
2      Mr  0.156673
3     Mrs  0.793651
4    Rare  0.250000
5   Royal  1.000000
"""
# 위 데이터를 바탕으로 1부터 6까지로 매핑을 숫자로 함 
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Royal':5, 'Rare':6}
for dataset in combine:
     dataset['Title'] = dataset['Title'].map(title_mapping)
     dataset['Title'] = dataset['Title'].fillna(0) #결측치가 있으면 0
train.head()

# Name과 PassengerId 삭제
train = train.drop(['Name', 'PassengerId'], axis =1)   # 한꺼번에 두개 삭제
test =  test.drop(['Name', 'PassengerId'], axis =1)   # 트레이닝에서 지우면 테스트에서도 반드시 지워라 
train.columns
test.columns      

combine = [train,test]
# 성별도 숫자로 치환
sex_mapping = {'male':0, 'female':1}
for dataset in combine:
     dataset['Sex'] = dataset['Sex'].map(sex_mapping)
train.head()

# 나이 age 가공
# age 에서 null  값에 대한 처리 
# 일단 -0.5 로 채워 넣은 후 pandas의 cut() 함수를 사용해서  AgeGroup를 만듬. 
# cut() 함수는 각 구간의 값을 특정값으로 정의해주는 함수 

import numpy as np
train['Age'] = train['Age'].fillna(-0.5)
test['Age'] = test['Age'].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ["Unknown", "Baby", "Child", "Teenager", "Student", "Young Adult", "Adult", "Senior"]
train['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
test['AgeGroup'] = pd.cut(test['Age'], bins, labels=labels)
train.head()
test.head()

bar_chart('AgeGroup')

#  {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Royal':5, 'Rare':6}
age_title_mapping == {1:'Young Adult', 2:'Young Adult', 3:'Adult', 4:'Adult', 5:'Adult', 6:'Adult'}
for i in range(len(train['AgeGroup'])):
   if train['AgeGroup'][i]=="Unknown":
       train['AgeGroup'][i] = age_title_mapping[train['Title'][i]]
for i in range(len(test['AgeGroup'])):
      if test['AgeGroup'][i]=="Unknown":
       test['AgeGroup'][i] = age_title_mapping[test['Title'][i]]
train.head()

# AgeGroup를 숫자로 치환 
age_mapping == {"Baby":1, "Child":2, "Teenager":3, "Student":4, "Young Adult":5, "Adult":6, "Senior":7}
train['AgeGroup'] =train['AgeGroup'].map(age_mapping)
test['AgeGroup'] =test['AgeGroup'].map(age_mapping)

train =train.drop(['Age'],axis=1)
test =test.drop(['Age'],axis=1)
train.head()

# Fare: 티켓의 요금
# qcut 함수를 사용. 4개의 범위로 cut 

train['FareBand'] =pd.qcut(train['Fare'],4,labels={1,2,3,4})
test['FareBand'] =pd.qcut(test['Fare'],4,labels={1,2,3,4})
train=train.drop(['Fare'],axis=1)
test=test.drop(['Fare'],axis=1)
train.head()

# *********************
# 데이터 모델링 
# **********************
train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape
# ((891, 8), (891,))






                     