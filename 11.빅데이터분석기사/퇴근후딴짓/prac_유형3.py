import numpy as np
import pandas as pd
from scipy import stats 
## https://www.kaggle.com/datasets/agileteam/bigdatacertificationkr/data 
'''
문제1. 다음은 22명의 학생들이 국어시험에서 받은 점수이다. 학생들의 평균이 75보다 크다고 할 수 있는가?
    귀무가설(H0): 모평균은 mu와 같다. (μ = mu), 학생들의 평균은 75이다
    대립가설(H1): 모평균은 mu보다 크다. (μ > mu), 학생들의 평균은 75보다 크다
    > 가정: 모집단은 정규분포를 따른다. & 표본의 크기가 충분히 크다.

검정통계량, p-value, 검정결과를 출력하시오
'''
scores = [75, 80, 68, 72, 77, 82, 81, 79, 70, 74, 76, 78, 81, 73, 81, 78, 75, 72, 74, 79, 78, 79]
stats.ttest_1samp(scores, 75, alternative='greater')
#TtestResult(statistic=np.float64(1.765879233231226), pvalue=np.float64(0.04597614747709146), 기각: 75보다 크다) 

'''
문제2. 어떤 특정 약물을 복용한 사람들의 평균 체온이 복용하지 않은 사람들의 평균 체온과 유의미하게 다른지 검정해보려고 합니다.
    약물을 복용한 그룹과 복용하지 않은 그룹의 체온 데이터가 각각 주어져 있다고 가정합니다.
    각 그룹의 체온은 정규분포를 따른다고 가정합니다.
검정통계량, p-value, 검정결과를 출력하시오
'''
group1 = [36.8, 36.7, 37.1, 36.9, 37.2, 36.8, 36.9, 37.1, 36.7, 37.1]
group2 = [36.5, 36.6, 36.3, 36.6, 36.9, 36.7, 36.7, 36.8, 36.5, 36.7]
stats.ttest_ind(group1, group2)
#TtestResult(statistic=np.float64(3.7964208654863336), pvalue=np.float64(0.001321891476703691), 기각:차이존재)

'''
문제3. 주어진 데이터는 고혈압 환자 치료 전후의 혈압이다. 해당 치료가 효과가 있는지 대응(쌍체)표본 t-검정을 진행하시오¶
귀무가설(H0): μ >= 0
대립가설(H1): μ < 0
    μ = (치료 후 혈압 - 치료 전 혈압)의 평균, 유의수준: 0.05

1) μ의 표본평균은?(소수 둘째자리까지 반올림) 
2) 검정통계량 값은?(소수 넷째자리까지 반올림)
3) p-값은?(소수 넷째자리까지 반올림)
4) 가설검정의 결과는? (유의수준 5%)
'''
df = pd.read_csv('data/high_blood_pressure.csv')
df.shape
df.head()

df['gap'] = df['bp_post'] - df['bp_pre']
df['gap'].mean()  # -6.17

stats.ttest_rel(df['bp_post'], df['bp_pre'], alternative='less')
#-3.0002, 0.0016, 기각: 효과가 있다 

'''
문4. 세 가지 다른 교육 방법(A, B, C)을 사용하여 수험생들의 시험 성적을 개선시키는 효과를 평가하고자 한다. 
30명의 학생들을 무작위로 세 그룹으로 배정하여 교육을 실시하였고, 시험을 보고 성적을 측정하였습니다. 
다음은 각 그룹의 학생들의 성적 데이터입니다.
    귀무가설(H0): 세 그룹(A, B, C) 간의 평균 성적 차이가 없다.
    대립가설(H1 또는 Ha): 세 그룹(A, B, C) 간의 평균 성적 차이가 있다.

다음 주어진 데이터로 일원배치법을 수행하여 그룹 간의 평균 성적 차이가 있는지 검정하세요 (f값, p값, 검정결과 출력) 
'''
groupA = [85, 92, 78, 88, 83, 90, 76, 84, 92, 87]
groupB = [79, 69, 84, 78, 79, 83, 79, 81, 86, 88]
groupC = [75, 68, 74, 65, 77, 72, 70, 73, 78, 75]

stats.f_oneway(groupA, groupB, groupC)
format(1.7529852237980142e-05, '.6f')
# 16.88, 0.000018, 기각: 차이가 있다. 

'''
문5. 12명의 수험생이 빅데이터 분석기사 시험에서 받은 점수이다. 
Shapiro-Wilk 검정을 사용하여 데이터가 정규 분포를 따르는지 검증하시오¶
    귀무 가설(H0): 데이터는 정규 분포를 따른다.
    대립 가설(H1): 데이터는 정규 분포를 따르지 않는다.
Shapiro-Wilk 검정 통계량, p-value, 검증결과를 출력하시오
'''
data = [75, 83, 81, 92, 68, 77, 78, 80, 85, 95, 79, 89]
stats.shapiro(data)
# 0.9768091723993144, 0.9676506711851194, 정규분포를 따른다 

'''
문6. iris에서 Sepal Length와 Sepal Width의 상관계수 계산하고 소수 둘째자리까지 출력하시오
'''
iris = pd.read_csv('data/iris.csv')
iris.corr(numeric_only=True)
print((round(iris['sepal_length'].corr(iris['sepal_width']),2))) # -0.12

'''
문7. Pclass, Gender, sibsp, parch를 독립변수로 사용하여 로지스틱 회귀모형을 실시하였을 때, 
parch변수의 계수값은? 단, Pclass는 범주형 변수이다 (반올림하여 소수 셋째 자리까지 계산)
'''
df = pd.read_csv('data/titanic.csv')
from statsmodels.formula.api import ols, logit 
model = logit('survived ~ pclass + sex + sibsp + parch', data=df).fit()
model.summary()   # -0.0503

'''
문8. 두 교육 방법의 효과 비교¶
연구자는 두 가지 다른 교육 방법이 학생들의 성적에 미치는 영향을 비교하고자 합니다. 
연구자는 무작위로 선발된 20명의 학생들을 두 그룹으로 나누어 한 그룹에는 교육 방법 A를, 다른 그룹에는 교육 방법 B를 적용합니다. 
교육이 끝난 후, 두 그룹의 성적을 비교하기 위해 독립 표본 t-검정과 ANOVA F-검정을 실시하려고 합니다.

다음은 두 그룹의 성적입니다: 다음의 두 가지 검정을 사용하여 두 교육 방법 간의 성적 차이가 통계적으로 유의한지를 검증하세요
    1.독립 표본 t-검정을 실시하여 t 통계량을 구하세요.
    2.독립 표본 t-검정을 실시하여 p-값을 구하세요.
    3.ANOVA F-검정을 실시하여 F 통계량을 구하세요.
    4.ANOVA F-검정을 실시하여 p-값을 구하세요.
'''
df = pd.DataFrame({
    'A':[77, 75, 82, 80, 81, 83, 84, 76, 75, 87],
    'B':[80, 74, 77, 79, 71, 74, 78, 69, 70, 72],
})
df

stats.ttest_ind(df['A'], df['B'])   # 3.1068522301122954, 0.006087373605949963
stats.f_oneway(df['A'], df['B'])    # 9.652530779753766, 0.0060873736059499145

'''
문9. 카이제곱 적합도 검정¶
고등학교에서는 졸업생들이 선택하는 대학 전공 분야의 선호도가 시간이 지남에 따라 변하지 않는다고 가정합니다. 
학교 측은 최근 졸업생들의 전공 선택이 과거와 같은 패턴을 따르는지 알아보기 위해 적합도 검정을 실시하기로 결정했습니다.

과거 자료에 따르면 졸업생들이 선택하는 전공의 분포는 다음과 같습니다:
    인문학: 20% 사회과학: 30% 자연과학: 25% 공학: 15% 기타: 10% 
올해 졸업한 학생 200명의 전공 선택 분포는 다음과 같았습니다:
    인문학: 30명 사회과학: 60명 자연과학: 50명 공학: 40명 기타: 20명 (total=200)

이 데이터를 바탕으로, 졸업생들의 전공 선택 패턴이 과거와 유사한지를 알아보기 위해 카이제곱 적합도 검정을 실시해야 합니다. 
유의 수준은 0.05로 설정합니다. (검정 통계량?, p-value?, 유의수준 하 귀무가설 기각 또는 채택?) 
'''
observed = [30, 60, 50, 40, 20]
expected = [200*0.2, 200*0.3, 200*0.25, 200*0.15, 200*0.1]
stats.chisquare(observed, expected)  # 5.833333333333334, 0.21194558437271782, 채택: 유사하다

'''
문10. 지지도, 신뢰도, 향상도¶
    지지도(A,B): A와 B가 함께 팔린 거래 횟수 / 전체 거래 횟수
    신뢰도(A->B): A와 B가 함께 팔린 거래 횟수 / A가 팔린 거래 횟수
    향상도(A,B): 신뢰도(A->B) / 지지도(B)
1. '빼빼로'와 '딴짓초코'가 함께 팔린 거래의 지지도를 계산하세요.
2. '빼빼로'가 팔린 거래 중에서 '빼빼로'와 '오징어칩'이 함께 팔린 거래의 신뢰도를 계산하세요.
3. '빼빼로'와 '양조위빵'의 향상도를 계산하세요.
'''
df = pd.DataFrame({
    'transaction': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    '빼빼로': [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
    '딴짓초코': [0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
    '양조위빵': [1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    '오징어칩': [0, 1, 1, 0, 0, 1, 0, 1, 1, 1],
    '초코파이': [1, 1, 0, 0, 1, 0, 1, 1, 0, 0]
})
df

len(df[(df['빼빼로']==1) & (df['딴짓초코']==1)]) / len(df)  # 0.2 
len(df[(df['빼빼로'] == 1) & (df['오징어칩'] ==1)]) / len(df[df['빼빼로'] == 1])  # 0.5714285714285714

x1 = len(df[(df['빼빼로'] == 1) & (df['양조위빵'] ==1)]) / len(df[df['빼빼로'] == 1])
x2 = len(df[(df['빼빼로']==1) & (df['양조위빵']==1)]) / len(df)
x1 / x2  # 1.4285714285714284

'''
문11. 포아송분포
한 서점에서는 평균적으로 하루에 3명의 고객이 특정 잡지를 구매합니다. 
이 데이터는 포아송 분포를 따른다고 가정할 때, 다음 질문에 대한 답을 구하세요.
(λ는 단위 시간(또는 단위 공간)당 평균 발생 횟수이고, k는 특정 시간(또는 공간) 동안의 이벤트 발생 횟수입니다.
 즉 λ = 3 (하루 평균 구매자 수), 𝑘는 관심 있는 구매자 수)

1.하루에 정확히 5명의 고객이 잡지를 구매할 확률은 얼마입니까? (%로 값을 정수로 입력하시오)
2.하루에 적어도 2명 이상의 고객이 잡지를 구매할 확률은 얼마입니까? (%로 값을 정수로 입력하시오)
'''
from scipy.stats import poisson 
poisson.pmf(5, mu=3)   # 0.10081881344492458 -> 10%
1- poisson.cdf(1, mu=3)    # 0.8008517265285442 -> 80%
## pmf : 확률 질량 함수(Poisson Mass Function), 특정 값 𝑘=5에 대한 확률을 반환
## cdf : 누적 분포 함수(Cumulative Distribution Function), 특정 값 𝑘=1 이하가 구매할 확률을 반환

'''
문12. 성별과 시험합격은 독립적인가를 검정하시오!
1 검정 통계량?
2 p-value?
3 귀무가설 기준 (기각/채택)?
4 남자의 합격 기대 빈도?
'''
data = pd.DataFrame({
    '남자':[100,200], 
    '여자':[130,170],    
    }, index = ['합격','불합격'])
data

stats.chi2_contingency(data)  # 5.929494712103407, 0.01488951060599475, 기각:서로 독립적이지 않다, 115명

'''
문12. 점 추정과 구간 추정¶
데이터셋은 어떤 도시의 일일 평균 온도 입니다.
    점추정: 데이터셋을 기반으로 이 도시의 평균 연간 온도를 점추정하세요. (반올림하여 소수 둘째자리까지)
    구간추정: 95% 신뢰수준에서 이 도시의 평균 연간 온도에 대한 신뢰구간을 구하세요. (반올림하여 소수 둘째자리까지)
'''
df = pd.read_csv('data/daily_temperatures.csv')
df.shape
df.head()

round(df.mean(), 2)   # 19.94

mu = df['Daily Average Temperature'].mean()
std = df['Daily Average Temperature'].std()
n = 365

inter_1 = mu-(1.96*std/np.sqrt(n))
inter_2 = mu+(1.96*std/np.sqrt(n))
inter_1, inter_2    # 19.42957247111869 ~ 20.445582616838387

'''
문13. 크리스마스 장식 종류와 지역에 따라 판매량에 유의미한 차이가 있는지 이원 분산 분석을 통해 검정하세요¶
    1.크리스마스 장식 종류(트리, 조명, 장식품)가 판매량에 미치는 영향을 분석하세요. 이때, 장식 종류의 F-value, p-value를 구하시오
    2.지역(북부, 남부, 동부, 서부)이 판매량에 미치는 영향을 분석하세요. 이때, 장식 종류의 F-value, p-value를 구하시오
    3.크리스마스 장식 종류와 지역의 상호작용이 판매량에 미치는 영향을 분석하세요. 이때, 장식 종류의 F-value, p-value를 구하시오
'''
df = pd.read_csv('data/christmas_decoration_sales.csv')
df.shape
df

from statsmodels.formula.api import ols, logit 
model = ols('Sales ~ C(Decoration_Type)*C(Region)', data=df).fit()
model.summary()

import statsmodels.api as sm
sm.stats.anova_lm(model) 
# 2.370578  0.114943 
# 0.720381  0.549614
# 2.308081  0.066915

'''
문13. 고객 정보를 나타낸 데이터이다. 주어진 데이터에서 500개 중 앞에서부터 350개는 train으로, 150개는 test 데이터로 나눈다. 
모델을 학습(적합)할 때는 train 데이터를 사용하고, 예측할 때는 test 데이터를 사용한다. 
모델은 로지스틱 회귀를 써서 고객이 특정 제품을 구매할지 여부를 예측하되, 페널티는 부과하지 않는다.
    종속변수: purchase (0: 구매 안 함, 1: 구매 함)

문제 1-1. income 변수를 독립변수로 purchase를 종속변수로 사용하여 로지스틱 회귀 모형을 만들고, 
income 변수가 한 단위 증가할 때 구매할 오즈비 값을 계산하시오. (반올림하여 소수 넷째자리까지 계산) : # 1.0000
문제 1-2. 독립변수 income만 사용해 학습한 모델에서 test 데이터의 purchase를 예측하고, accuracy (정확도)를 구하시오. 
(반올림하여 소수 셋째자리까지 계산) : 0.507
문제 1-3. 독립변수 income만 사용해 학습한 모델의 로짓 우도를 계산하시오. : -242.41
문제 1-4. 독립변수 income만 사용해 학습한 모델의 유의확률(p-value)를 구하시오. : 0.5964
'''

df = pd.read_csv('data/Customer_Data.csv')
df.shape
df.head()
df.info()
df.isnull().sum()

train = df[:350]
test = df[350:]

from statsmodels.formula.api import ols, logit 
model = logit('purchase ~ income', data=train).fit()
model.summary()
round(np.exp(1.96e-06),4)  # 1.0000

from sklearn.metrics import accuracy_score
pred = model.predict(test)
pred = (pred >= 0.5).astype(int)
acc = accuracy_score(test['purchase'], pred)
acc

'''
문14.앞의 고객 정보 데이터로 부터 age, income, marital_status 변수를 독립변수로 purchase를 종속변수로 사용하여 
로지스틱 회귀 모형을 만들고, 잔차이탈도를 구하시오. (반올림하여 소수 넷째자리까지 계산) : 691.4035
** 잔차이탈도 = -2 x llf 
'''
from statsmodels.formula.api import ols, logit
model = logit('purchase ~ age + income + marital_status', data=df).fit()
model.summary()

-2 * model.llf

'''
문15. 로지스틱 회귀 모델의 AIC와 BIC 계산¶
1. 독립변수(Pregnancies, BMI, DiabetesPedigreeFunction)를 사용하여 로지스틱 회귀 모델을 학습하고, 
학습된 모델의 AIC와 BIC 값을 계산하세요.(종속변수는 Outcome)
2. Pregnancies를 범주형 변수로 처리하여 로지스틱 회귀 모델을 학습하세요. 
(독립변수(Pregnancies, BMI, DiabetesPedigreeFunction)를 사용) 모델의 로그 우도(Log-Likelihood)를 구하라
'''
df = pd.read_csv('data/diabetes_train.csv')
df.shape
df.head()
df.info()

from statsmodels.formula.api import ols, logit 
model = logit('Outcome ~ Pregnancies + BMI + DiabetesPedigreeFunction', data=df).fit()
model.summary()
model.aic, model.bic   # np.float64(640.8879324373285), np.float64(658.312363080112)

model = logit('Outcome ~ C(Pregnancies) + BMI + DiabetesPedigreeFunction', data=df).fit()
model.summary()
model.llf  #-304.35

'''
문16. 
1)모든 변수를 사용하여 OLS 모델을 적합하고, 회귀계수 중 가장 큰 값은? : x1 ~ 1.9979
2)유의미하지 않은 변수를 제거한 후 모델을 다시 적합하고, 회귀계수 중 가장 작은 변수명은? : x2 ~ -1.4942
3)2번 모델의 R-squared 값을 계산하고 해석하세요. : 0.988
4)1번 모델에서 새로운 데이터(x1=5, x2=12, x3=10, x4=3)에 대해 y 값을 예측하세요. : -0.243308
5) 1번 모델에서 x1, x2, x3, x4의 상관관계를 계산하고 가장 큰 상관계수를 구하시오. (단, 자기 상관관계 제외) : x2 & x4 = -0.224881
6) x1과 x2만을 예측 변수로 사용하는 모델을 적합하고, 전체 모델과 R-squared 값을 구하시오. : 0.964
7) 잔차(residual) 분석을 수행하고, 잔차의 표준편차를 구하시오. : 0.9001841451852483
8) 1번 모델에서 새로운 데이터(x1=5, x2=12, x3=10, x4=3)에 대해 y의 신뢰구간 하한(97% 신뢰수준)을 구하세요. : -0.540231
9) 1번 모델에서 새로운 데이터(x1=5, x2=12, x3=10, x4=3)에 대해 y의 예측구간 상한(97% 신뢰수준)을 구하세요. : 1.802934
'''
df = pd.read_csv('data/t3_regression_data.csv')
df.shape
df.head()

from statsmodels.formula.api import ols, logit 
import statsmodels.api as sm 

model_1 = ols('y ~ x1 + x2 + x3 + x4', data=df).fit()
model_1.summary()

model_2 = ols('y ~ x1 + x2 + x3', data=df).fit()
model_2.summary()

new = pd.DataFrame({
    'x1':[5], 'x2':[12], 'x3':[10], 'x4':[3]
})
model_1.predict(new)

df.corr()

model_3 = ols('y ~ x1 + x2', data=df).fit()
model_3.summary()
#잔차
model_1.resid.std()
#신뢰/예측 구간
model_1.get_prediction(new).summary_frame(alpha=0.03)
'''
>>> model_1.get_prediction(new).summary_frame(alpha=0.03)
       mean  mean_se  mean_ci_lower  mean_ci_upper  obs_ci_lower  obs_ci_upper
0 -0.243308  0.13477      -0.540231       0.053614      -2.28955      1.802934
'''

'''
문16. 베스킨라빈스는 쿼트(Quart) 아이스크림의 중앙값이 620g이라고 주장하고 있습니다. 
저는 실제로 이 아이스크림의 중앙값이 620g보다 무겁다고 주장합니다. 다음은 20개의 쿼트 아이스크림 샘플의 무게 측정 결과입니다. 
이 측정 결과를 바탕으로 나의 주장이 사실인지 비모수 검정(Wilcoxon Signed-Rank Test)을 통해 검정해보십시오. 
p-value값을 반올림하여 소수점 둘째 자리까지 계산
    귀무가설: "베스킨라빈스 쿼트 아이스크림의 중앙값은 620g이다."
    대립가설: "베스킨라빈스 쿼트 아이스크림의 중앙값은 620g보다 무겁다."
'''
data = {
    "weight": [630, 610, 625, 615, 622, 618, 623, 619, 620, 624, 616, 621, 617, 629, 626, 620, 618, 622, 625, 615, 
               628, 617, 624, 619, 621, 623, 620, 622, 618, 625, 616, 629, 620, 624, 617, 621, 623, 619, 625, 618,
               622, 620, 624, 617, 621, 623, 619, 625, 618, 622]
}
df = pd.DataFrame(data)
df

from scipy.stats import wilcoxon 
wilcoxon(df['weight'], 620, alternative='greater')
# 0.03 --> 귀무가설 기각 : 620g보다 무겁다 

