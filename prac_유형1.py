import numpy as np 
import pandas as pd
## https://www.kaggle.com/datasets/agileteam/bigdatacertificationkr/data 
'''
q1. 데이터에서 IQR을 활용해 Fare컬럼의 이상치를 찾고, 이상치 데이터의 여성 수를 구하시오
'''
df = pd.read_csv('data/titanic.csv')
df.head()
df.info()

iqr = df['fare'].quantile(0.75) - df['fare'].quantile(0.25)
lower_lim = df['fare'].quantile(0.25) - (1.5 * iqr)
upper_lim = df['fare'].quantile(0.75) + (1.5 * iqr)

df_n = df[
    (df['fare'] < lower_lim) | (df['fare'] > upper_lim) 
]
df_n['sex'].value_counts()   # female    70

'''
q2. 주어진 데이터에서 이상치(소수점 나이)를 찾고 올림, 내림, 버림(절사)했을때 3가지 모두 
이상치 'age' 평균을 구한 다음 모두 더하여 출력하시오
'''
df = pd.read_csv('data/basic1.csv')
df.head()
df.info()

df_n = df[df['age']%1 != 0]
df_n

x1 = np.ceil(df_n['age'])
x2 = np.floor(df_n['age'])
x3 = np.trunc(df_n['age'])
x1.mean() + x2.mean() + x3.mean()    # 69.5

'''
q3. 주어진 데이터에서 결측치가 80%이상 되는 컬럼은(변수는) 삭제하고, 80% 미만인 결측치가 있는 컬럼은 
'city'별 중앙값으로 값을 대체하고 'f1'컬럼의 평균값을 출력하세요!
'''
df = pd.read_csv('data/basic1.csv')
df.isnull().sum()

df = df.drop('f3',axis=1)

x_서울 = df[df['city'] == '서울']['f1'].median()
x_부산 = df[df['city'] == '부산']['f1'].median()
x_대구 = df[df['city'] == '대구']['f1'].median()
x_경기 = df[df['city'] == '경기']['f1'].median()

df['f1'] = df['f1'].fillna(df['city'].map({
    '서울':x_서울, 
    '부산':x_부산, 
    '대구':x_대구, 
    '경기':x_경기, 
}))

df['f1'].mean()   # 65.52

'''
q4. 주어진 데이터 중 train.csv에서 'SalePrice'컬럼의 왜도와 첨도를 구한 값과, 
'SalePrice'컬럼을 스케일링(log1p)로 변환한 이후 왜도와 첨도를 구해 모두 더한 다음 소수점 2째자리까지 출력하시오
'''
df = pd.read_csv('data/train.csv')
df.head()
df.info()

a = df['SalePrice'].skew() + df['SalePrice'].kurt()   # 8.419157619832742
df['SalePrice'] = np.log1p(df['SalePrice'])
b = df['SalePrice'].skew() + df['SalePrice'].kurt()   # 0.9308657756047314
round(a+b,2)    # 1.81

'''
q5. 주어진 데이터 중 basic1.csv에서 'f4'컬럼 값이 'ENFJ'와 'INFP'인 'f1'의 표준편차 차이를 절대값으로 구하시오
'''
df = pd.read_csv('data/basic1.csv')
df_enfj = df[df['f4'] == 'ENFJ']
df_infp = df[df['f4'] == 'INFP']
abs(df_enfj['f1'].std() - df_infp['f1'].std())     # 5.859621525876811

'''
q6. 주어진 데이터 중 basic1.csv에서 'f1'컬럼 결측 데이터를 제거하고, 
'city'와 'f2'을 기준으로 묶어 합계를 구하고, 'city가 경기이면서 f2가 0'인 조건에 만족하는 f1 값을 구하시오
'''
df = df.dropna(subset='f1')
df_g = df.groupby(['city','f2']).sum().reset_index()
res = df_g[
    (df_g['city'] =='경기') & (df_g['f2'] == 0)
]
res.T   # 833.0

'''
q7. 'f4'컬럼의 값이 'ESFJ'인 데이터를 'ISFJ'로 대체하고, 'city'가 '경기'이면서 'f4'가 'ISFJ'인 데이터 중 
'age'컬럼의 최대값을 출력하시오!
'''
df = pd.read_csv('data/basic1.csv')
df['f4'] = df['f4'].replace('ESFJ','ISFJ')
df_n = df[
    (df['city'] == '경기') & (df['f4'] == 'ISFJ')
]
df_n['age'].max()     # 90.0

'''
q8. 주어진 데이터 셋에서 'f2' 컬럼이 1인 조건에 해당하는 데이터의 'f1'컬럼 누적합을 계산한다. 
이때 발생하는 누적합 결측치는 바로 뒤의 값을 채우고, 누적합의 평균값을 출력한다. 
(단, 결측치 바로 뒤의 값이 없으면 다음에 나오는 값을 채워넣는다)
'''
df = pd.read_csv('data/basic1.csv')
df_n = df[df['f2'] == 1]
res = df_n['f1'].cumsum()
res = res.fillna(method='bfill')
res.mean()    #980.3783783783783

'''
q9. 주어진 데이터에서 'f5'컬럼을 표준화(Standardization (Z-score Normalization))하고 그 중앙값을 구하시오
'''
df = pd.read_csv('data/basic1.csv')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['f5'] = scaler.fit_transform(df[['f5']])
df['f5'].median()       # 0.260619629559015

'''
q11. 주어진 데이터에서 'f5'컬럼을 min-max 스케일 변환한 후, 상위 5%와 하위 5% 값의 합을 구하시오
'''
df = pd.read_csv('data/basic1.csv')
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler() 
df['f5'] = scaler.fit_transform(df[['f5']])
df['f5'].quantile(0.95) + df['f5'].quantile(0.05)    # 1.0248740983597389

'''
q12. 주어진 데이터에서 상위 10개 국가의 접종률 평균과 하위 10개 국가의 접종률 평균을 구하고, 그 차이를 구해보세요 
(단, 100%가 넘는 접종률 제거, 소수 첫째자리까지 출력) *국가별 최종 접종률을 구해야 함을 주의할 것 
'''
df = pd.read_csv('data/covid-vaccination-vs-death_ratio.csv')
df.shape
df.head()
df.info()

g = df.groupby('country')['ratio'].max().reset_index()
g

g = g[g['ratio'] <= 100]
top10 = g.sort_values(by='ratio', ascending=False)[:10]
bottom10 = g.sort_values(by='ratio')[:10]
round(top10['ratio'].mean() - bottom10['ratio'].mean() , 1)     # 88.4

'''
q13. 주어진 데이터에서 상관관계를 구하고, quality와의 상관관계가 가장 큰 값과, 가장 작은 값을 구한 다음 더하시오!
단, quality와 quality 상관관계 제외, 소수점 둘째 자리까지 반올림하여 계산
'''
df = pd.read_csv('data/winequality-red.csv', sep=';')
df.head()
df.info()

abs(df.corr()['quality']).sort_values()   # 0.013732 + 0.476166 = 0.489898

'''
q14. city와 f4를 기준으로 f5의 평균값을 구한 다음, 
f5를 기준으로 상위 7개 값을 모두 더해 출력하시오 (소수점 둘째자리까지 출력)
'''
df = pd.read_csv('data/basic1.csv')
g = df.groupby(['city','f4'])['f5'].mean().reset_index()
res = g.sort_values(by='f5', ascending=False)[:7]
round(res['f5'].sum(), 2)    #643.68 

'''
q15. 주어진 데이터 셋에서 age컬럼 상위 20개의 데이터를 구한 다음 
f1의 결측치를 중앙값으로 채운다. 그리고 f4가 ISFJ와 f5가 20 이상인 f1의 평균값을 출력하시오!
'''
df = pd.read_csv('data/basic1.csv')
df_n = df.sort_values(by='age', ascending=False)[:20]
df_n['f1'] = df_n['f1'].fillna(df_n['f1'].median())
res = df_n[
    (df_n['f4'] == 'ISFJ') & (df_n['f5'] >= 20)
]
res['f1'].mean()    # 73.875

'''
q16. 주어진 데이터 셋에서 f2가 0값인 데이터를 age를 기준으로 오름차순 정렬하고 앞에서 부터 20개의 데이터를 추출한 후 
f1 결측치(최소값)를 채우기 전과 후의 분산 차이를 계산하시오 (소수점 둘째 자리까지)
'''
df = pd.read_csv('data/basic1.csv')
df_n = df[df['f2'] == 0]
res = df_n.sort_values(by='age')[:20]
res['f1'].var() # 351.7636363636363

res['f1'] = res['f1'].fillna(res['f1'].min())
res['f1'].var() # 313.3263157894737

round(351.7636363636363 - 313.3263157894737, 2 )   # 38.44

'''
q17. 2022년 5월 sales의 중앙값을 구하시오
'''
df = pd.read_csv('data/basic2.csv')
df.shape
df.head()

df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df_n = df[
    (df['year'] == 2022) & (df['month'] == 5)
    ]
df_n['Sales'].median()    # 1477685.0

'''
q18. 주어진 데이터에서 2022년 5월 주말과 평일의 sales컬럼 평균값 차이를 절대값으로 구하시오 
(소수점 둘째자리까지 출력, 반올림)
'''
df = pd.read_csv('data/basic2.csv')
df['Date'] = pd.to_datetime(df['Date'])

df['yy-mm'] = df['Date'].dt.to_period('M')
df['dayofweek'] = df['Date'].dt.day_name()

df['tag'] = df['dayofweek'].map({
    'Monday': 0, 'Tuesday': 0, 'Wednesday': 0, 'Thursday': 0, 'Friday': 0, 
    'Saturday': 1, 'Sunday': 1,
})
df

df_n = df[df['yy-mm'] == '2022-05']
res1 = df_n[df_n['tag'] == 1]['Sales'].mean()
res2 = df_n[df_n['tag'] == 0]['Sales'].mean()
round(abs(res1 - res2), 2)      # 3010339.1

'''
q19. 주어진 데이터에서 2022년 월별 Sales 합계 중 가장 큰 금액과 2023년 월별 Sales 합계 중 가장 큰 금액의 차이를 
절대값으로 구하시오. 단 Events컬럼이 '1'인경우 80%의 Salse값만 반영함 (최종값은 소수점 반올림 후 정수 출력)
'''
df = pd.read_csv('data/basic2.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['Sales'] = np.where(df['Events'] == 1, df['Sales']*0.8, df['Sales'])

df_2022 = df[df['year'] == 2022]
res_2022 = df_2022.groupby('month')['Sales'].sum().reset_index()

df_2023 = df[df['year'] == 2023]
res_2023 = df_2023.groupby('month')['Sales'].sum().reset_index()

round(abs(res_2022.max() - res_2023.max()),0)   #42473436

'''
q20. basic1 데이터와 basic3 데이터를 basic1의 'f4'값을 기준으로 병합하고,
병합한 데이터에서 r2결측치를 제거한다음, 앞에서 부터 20개 데이터를 선택하고 'f2'컬럼 합을 구하시오
'''
df1 = pd.read_csv('data/basic1.csv')
df3 = pd.read_csv('data/basic3.csv')
df1.shape, df3.shape
df1.head()
df3.head()

df = pd.merge(df1, df3, on='f4', how='inner')
df = df.dropna(subset='r2')
df[:20]['f2'].sum()     # 15 

'''
q21. basic1 데이터 중 'age'컬럼 이상치를 제거하고, 동일한 개수로 나이 순으로 3그룹으로 나눈 뒤 
각 그룹의 중앙값을 더하시오 (이상치는 음수(0포함), 소수점 값)
'''
df = pd.read_csv('data/basic1.csv')
df = df[
    (df['age'] %1 == 0) & (df['age'] > 0)
]

df_n = df.sort_values(by='age')
df1 = df_n[:30]
df2 = df_n[30:60]
df3 = df_n[60:]

df1['age'].median() + df2['age'].median() + df3['age'].median()    # 165.0

'''
q22. 주어진 데이터(basic2.csv)에서 주 단위 Sales의 합계를 구하고, 
가장 큰 값을 가진 주와 작은 값을 가진 주의 차이를 구하시오(절대값)
'''
df = pd.read_csv('data/basic2.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['week'] = df['Date'].dt.to_period('W')

df_g = df.groupby('week')['Sales'].sum().reset_index()
df_g.sort_values(by='Sales', ascending=False)
# 99165648 - 7526598 = 91639050

'''
q23. f1의 결측치를 채운 후 age 컬럼의 중복 제거 전과 후의 'f1' 중앙값 차이를 구하시오¶
- 결측치는 'f1' 데이터에서 큰 값 순으로 정렬했을 때 10번째에 위치한 값으로 채운다.
- 중복 데이터 발생시 뒤에 나오는 데이터를 삭제함
- 최종 결과값은 절대값으로 출력
 (중복 제거 기준: 제공된 데이터 순서대로 중복 데이터 발생시 뒤에 나오는 데이터를 삭제함)
'''
df = pd.read_csv('data/basic1.csv')
x = df.sort_values(by='f1', ascending=False)['f1'].iloc[9]
df['f1'] = df['f1'].fillna(x)

x1 = df['f1'].median()
df = df.drop_duplicates(subset='age', keep='first')
x2 = df['f1'].median()
abs(x1 - x2)   # 0.5 

'''
q24. 주어진 데이터(basic2.csv)에서 "pv"컬럼으로 1일 시차(lag)가 있는 새로운 컬럼을 만들고
(예: 1월 2일에는 1월 1일 pv데이터를 넣고, 1월 3일에는 1월 2일 pv데이터를 넣음),
새로운 컬럼의 1월 1일은 다음날(1월 2일)데이터로 결측치를 채운 다음, 
Events가 1이면서 Sales가 1000000이하인 조건에 맞는 새로운 컬럼 합을 구하시오
'''
df = pd.read_csv('data/basic2.csv')
df.head()

df['PV_lag'] = df['PV'].shift(1)
df.loc[0,'PV_lag'] = df.loc[1,'PV_lag']

df = df[
    (df['Events'] == 1) & (df['Sales'] <= 1000000)
]
df['PV_lag'].sum()    # 1894876.0

'''
q25. basic1 데이터에서 f4가 E로 시작하면서 부산에 살고 20대인 사람은 몇 명일까요?
'''
df = pd.read_csv('data/basic1.csv')
df.head()

df = df[
    (df['f4'].str[:1] == 'E') & (df['city'] == '부산') & (df['age'] >= 20) & (df['age'] < 30)
]
# 0명

'''
q26. menu컬럼에 "라떼" 키워드가 있는 데이터의 수는?
'''
df = pd.read_csv('data/payment.csv')
df['menu'].str.contains('라떼').sum()

'''
q27. 바닐라라떼 5점, 카페라떼 3점, 아메리카노 2점, 나머지 0점이다 총 메뉴의 점수를 더한 값은?
'''
df = pd.read_csv('data/payment.csv')
df['menu'].unique()
df['menu'] = df['menu'].str.replace(' ', '')
df

df['score'] = df['menu'].map({
    '바닐라라떼':5, '카페라떼':3, '아메리카노':2 
})
df['score'] = df['score'].fillna(0)
df['score'].sum()    # 17

'''
q28. 시간(hour)이 13시 이전(13시 포함하지 않음) 데이터 중 가장 많은 결제가 이루어진 날짜(date)는? 
(date 컬럼과 동일한 양식으로 출력)¶
'''
df = pd.read_csv('data/payment.csv')
df_13 = df[df['hour'] < 13 ]
df_13.sort_values('date')   # 20221203

'''
q29. 12월인 데이터 수는?
'''
df = pd.read_csv('data/payment.csv')
df.head()
df.info()
df['date'] = df['date'].astype(str)
df['month'] = df['date'].str[4:6]

df[df['month'] == '12'].shape   # 11 

'''
q30. 12월 25일 결제 금액(price)은 12월 총 결제금액의 몇 %인가? (정수로 출력)
'''
df = pd.read_csv('data/payment.csv')
df['date'] = df['date'].astype(str)
df['month'] = df['date'].str[4:6]
df_12 = df[df['month'] == '12']

x1 = df_12[df_12['date'] == '20221225']['price'].sum()
x2 = df_12['price'].sum()
(x1 / x2)*100    # 26

'''
q31. 수학, 영어, 국어 점수 중 사람과 과목에 상관없이 가장 상위 점수 5개를 모두 더하고 출력하시오.
'''
df = pd.DataFrame({'Name': {0: '김딴짓', 1: '박분기', 2: '이퇴근'},
                   '수학': {0: 90, 1: 93, 2: 85},
                   '영어': {0: 92, 1: 84, 2: 86},
                   '국어': {0: 91, 1: 94, 2: 83},})

df

res = pd.melt(df, id_vars = ['Name'])
res = res.sort_values(by='value', ascending=False)
res['value'][:5].sum()    # 460 

'''
q32. 수학, 영어 점수 중 사람과 과목에 상관없이 90점 이상인 점수의 평균을 정수로 구하시오 (소수점 버림)
'''
df = pd.DataFrame({'Name': {0: '김딴짓', 1: '박분기', 2: '이퇴근'},
                   '수학': {0: 90, 1: 93, 2: 85},
                   '영어': {0: 92, 1: 84, 2: 86},
                   '국어': {0: 91, 1: 94, 2: 83},})

df

res = pd.melt(df, id_vars=['Name'], value_vars=['수학','영어'] )
res = res[res['value'] >= 90]
res['value'].mean()    # 91

'''
q33. 
1) 세션의 지속 시간을 분으로 계산하고 가장 긴 지속시간을 출력하시오(반올림 후 총 분만 출력) : 300분
2) 가장 많이 방문한 Page를 찾고 그 페이지의 머문 평균 시간을 구하시오 (반올림 후 총 시간만 출력) : 3시간 
3) 사용자들이 가장 활발히 활동하는 시간대(예: 새벽, 오전, 오후, 저녁)를 분석하세요. 
   이를 위해 하루를 4개의 시간대로 나누고 각 시간대별로 세션의 수를 계산하고, 그 중에 가장 많은 세션 수를 출력 : 새벽 2447
    - 새벽: 0시 부터 6시 전 / 오전: 6시 부터 12시 전 / 오후: 12 부터 18시 전 / 저녁: 18시 부터 0시 전
4) user가 가장 많이 접속 했던 날짜를 출력하시오. (예, 2023-02-17) : 2023-03-28
'''
df = pd.read_csv('data/website.csv')
df.info()
df.head()

df['StartTime'] = pd.to_datetime(df['StartTime'])
df['EndTime'] = pd.to_datetime(df['EndTime'])
#1
df['duration'] = (df['EndTime'] - df['StartTime']).dt.total_seconds()/60
df.sort_values(by='duration', ascending=False)
#2
df['Page'].value_counts()
round((df[df['Page'] == 'Page5']['duration'].mean()) / 60, 0)
#3
df['s_hour'] = df['StartTime'].dt.hour 

def time_slot(hour): 
    if 0 <= hour < 6 : 
        return '새벽'
    elif 6 <= hour < 12 : 
        return '오전'
    elif 12 <= hour < 18 : 
        return '오후'
    else : 
        return '저녁'
    
df['timeslot'] = df['s_hour'].apply(time_slot)
df['timeslot'].value_counts()   # 새벽    2447
#4
df['s_day'] = df['StartTime'].dt.to_period('D')
df_g = df.groupby('s_day')['UserID'].size().reset_index()
df_g.sort_values('UserID', ascending=False)   # 2023-03-28

'''
q34. 
1) 사용자별 방문 패턴 분석: 각 사용자가 방문한 페이지별로 평균 세션 지속 시간을 계산하고, 
   각 페이지별로 가장 긴 평균 세션 시간을 가진 사용자를 찾으세요. 
   이를 위해 각 사용자의 페이지별 평균 세션 시간을 계산하고, 각 페이지에서 가장 긴 평균 세션 시간을 모두 더하고 
   정수형으로 출력하시오.

2) 시간대별 페이지 선호도 분석: 하루를 네 시간대로 나누고 (새벽: 0-6시, 오전: 6-12시, 오후: 12-18시, 저녁: 18-24시), 
   각 시간대별로 가장 많이 방문된 페이지를 찾으세요. 각 시간대별로 가장 많이 방문된 페이지의 이름과 해당 시간대의 
   방문 횟수를 찾으세요! (단 0-6시 일때 0시부터 6시 전까지입니다.)
    - 2-1: 시간대 별로 나누었을 때 가장 많이 방문한 페이지의 이름을 구하시오.
    - 2-2: 시간대 별로 나누었을 때 가장 방문 횟수가 큰 값을 구하시오

3) 재방문 패턴 분석: 사용자가 같은 날짜에 여러 페이지를 방문하는 경우를 '재방문'으로 간주합니다. 
   재방문한 사용자들의 데이터를 분석하여, 재방문한 날짜별 총 방문 페이지 수를 구하고 가장 재방문이 많은 월을 정수로 구하시오
   (예를 들어, 사용자가 2024년 6월 20일에 Page1을 두 번 방문하거나, 한 번에 Page1과 Page2를 방문한 경우 모두 재방문으로 처리)
'''
df = pd.read_csv('data/website.csv')
df.info()

df['StartTime'] = pd.to_datetime(df['StartTime'])
df['EndTime'] = pd.to_datetime(df['EndTime'])

#1
df['duration(시간)'] = ((df['EndTime'] - df['StartTime']).dt.total_seconds()) / 60 / 60
df_g = df.groupby(['Page','UserID'])['duration(시간)'].mean().reset_index()
df_g

max_ids = df.groupby('Page')['duration(시간)'].idxmax()
max_ids
df_g.iloc[max_ids]['duration(시간)'].sum()

#2
df['s_hour'] = df['StartTime'].dt.hour

def time_slot(hour): 
    if 0 <= hour < 6: 
        return '새벽'
    elif 6 <= hour < 12: 
        return '오전'
    elif 12 <= hour < 18: 
        return '오후'
    else: 
        return '저녁'

df['시간대'] = df['s_hour'].apply(time_slot)
df_g = df.groupby(['시간대','Page']).size().reset_index()
df_g  #  새벽  Page4  526, 오전  Page5  538,  오후  Page3  472, 저녁  Page5  512

#3
df['s_day'] = df['StartTime'].dt.date

df_g = df.groupby(['UserID','s_day'])['Page'].count().reset_index()
res = df_g[df_g['Page'] >1]
res    

res['s_day'] = pd.to_datetime(res['s_day'])
res['month'] = res['s_day'].dt.month
res = res.groupby('month')['Page'].sum().reset_index()
res.sort_values(by='Page', ascending=False)  # 4월 

'''
q34. 
1) 각 피드백 중에서 가장 긴 피드백을 작성한 UserID를 찾고 해당 UserID가 주문을 몇 번 했는지 구하시오
2) 주어진 데이터에서 '제품'이라는 단어가 가장 많이 포함된 카테고리(그룹)을 찾고, 
   해당 카테고리에 속한 피드백들의 평균 배송 시간(분)을 구하시오. 
'''
df = pd.read_csv('data/e-commerce.csv')
df.shape
df.head()
df.info()

#1
df['fb_length'] = df['Feedback'].str.len()
df.sort_values(by='fb_length', ascending=False).head(1).T   # UserID  8
len(df[df['UserID'] == 8])    # 3건 

#2 
df['tag'] = df['Feedback'].str.contains('제품')
df_g = df.groupby('Category')['tag'].sum().reset_index()
df_g   #  서비스    6

df_service = df[df['Category'] == '서비스']
df_service['OrderDate'] = pd.to_datetime(df_service['OrderDate'])
df_service['ArrivalDate'] = pd.to_datetime(df_service['ArrivalDate'])
df_service['배송시간(분)'] = (df_service['ArrivalDate'] - df_service['OrderDate']).dt.total_seconds()/60
df_service['배송시간(분)'].mean()   # 9048.75분

