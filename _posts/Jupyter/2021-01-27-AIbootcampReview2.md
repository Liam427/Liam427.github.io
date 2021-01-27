---
title: " AI Bootcamp Section1 Review2"
layout: single
use_math: true
author_profile: true
read_time: true
comments: true
share: true
related: true
categories:
    - AI Bootcamp
tag:
    - AI Bootcamp
    - Diary
    - Post
    - Codestates
---

<p align="center">
  <img src="/assets/img/post/AIbootcamp.jpg" alt="AI Bootcamp"/>
</p>  

# AI Bootcamp Section1 Review  
---  
### 2. Statistics(Hypothesis Test)

* week2에서는 통계학을 배웠는데 그 중에서도 가설검정 하는걸 위주로 배웠기 때문에 소제목을 Hypothesis Test라고 지었다.  
    그중에서도 특히 One-sample t-test, Two-sample t-test, anova, One-sample Chi-square test, Two-sample Chi-square test, Bayesian  
    정도가 중요하다고 생각이 들기 때문에 그중에서도 이해가 잘 가지 않은 부분에 대해 포스팅 하겠다.

#### **1) One-sample t-test**

* 전체 학생들 중 20명의 학생들을 추려 키를 재서 전체 학생들의 평균 키가 175cm인지 아닌지 알아보고 싶다.  
    $H_0$ : 학생들의 평균 키가 175cm이다.  
    $H_1$ : 학생들의 평균 키가 175cm가 아니다.  


```python
import numpy as np
from scipy import stats
 
#to get consistent result
np.random.seed(1)
 
#generate 20 random heights with mean of 180, standard deviation of 5
heights= [180 + np.random.normal(0,5)for _ in range(20)]
 
#perform 1-sample t-test
tTestResult1 = stats.ttest_1samp(heights,175)

#histogram
import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(heights, kde=False, fit=sp.stats.norm)
plt.show();
 
#print result
print("The T-statistic is %.3f and the p-value is %.3f" % tTestResult1)
```


<p align="center">
  <img src="/images/2021-01-27-AIbootcampReview2_files/2021-01-27-AIbootcampReview2_4_0.png" alt="Review2"/>
</p>  
    
    


    The T-statistic is 3.435 and the p-value is 0.003
    

* p-value 가 0.003으로, 기각역을 p < 0.05로 설정했을 때 귀무 가설을 기각한다.  
    즉, 귀무 가설이 참일때 (학생들의 실제 평균 키가 175cm일때) 위와 같은 표본을 얻을 확률이 0.003으로,  
    학생들의 평균 키는 175cm가 아니라고 할 수 있다.



#### **2) Two-sample t-test**


```python
#perform 2-sample t-test
tTestResult2= stats.ttest_ind(heights,175)

#histogram
import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(heights, kde=False, fit=sp.stats.norm)
plt.show();
 
#print result
print("The T-statistic is %.3f and the p-value is %.3f" % tTestResult2)
```


    
<p align="center">
  <img src="/images/2021-01-27-AIbootcampReview2_files/2021-01-27-AIbootcampReview2_7_0.png" alt="Review2"/>
</p>
    


    The T-statistic is nan and the p-value is nan
    

#### **3) ANOVA**

**두 개 이상의 집단에 대해 평균비교를 하고자 할 때 기존의 t-test를 사용한다면, 두 집단씩 짝을 지어 t-test를 진행해야 함**  
* 세 개의 집단이 있을 때, 둘씩 짝을 짓는 경우의 수: 3가지  
* 네 개의 집단이 있을 때, 둘씩 짝을 짓는 경우의 수: 6가지  
* 다섯 개의 집단이 있을 때, 둘씩 짝을 짓는 경우의 수: 10가지  
* 여섯 개의 집단이 있을 때, 둘씩 짝을 짓는 경우의 수: 15가지  
* t-test로만 진행한다면, 분석횟수가 기하급수적으로 증가함   
    ⇒ 과잉검증의 문제가 발생함

**ANOVA에서 사용하는 용어**  
* 요인(factor): 집단을 구별하는 (독립)변수를 분산분석의 맥락에서는 "요인"이라고 칭함. 예) 성별, 국가
* 수준(level): 요인의 수준. 즉, 각 집단을 의미함. 예) 요인이 "성별"일 때, 수준은 "남", "여"
* 상호작용: 한 요인의 수준에 따른 종속변수의 차이가 또 다른 요인의 수준에 따라 달라질 때, "요인들 간 상호작용이 존재한다"고 함

**일원분산분석**  

$H_0$ : 모든 집단의 평균이 동일하다.  
$H_1$ : 적어도 한 집단의 평균이 다른 집단들과 다르다.  


```python
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/Liam427/stuydy-data/main/data/PlantGrowth.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight</th>
      <th>group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.17</td>
      <td>ctrl</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.58</td>
      <td>ctrl</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.18</td>
      <td>ctrl</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.11</td>
      <td>ctrl</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.50</td>
      <td>ctrl</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.group.unique()
```




    array(['ctrl', 'trt1', 'trt2'], dtype=object)




```python
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

model = ols('weight ~ C(group)', df).fit()
anova_lm(model)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>df</th>
      <th>sum_sq</th>
      <th>mean_sq</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C(group)</th>
      <td>2.0</td>
      <td>3.76634</td>
      <td>1.883170</td>
      <td>4.846088</td>
      <td>0.01591</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>27.0</td>
      <td>10.49209</td>
      <td>0.388596</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



* Pr(>F)가 p-value. 이 값이 0.05보다 작으면 통계적으로 유의미한 차이가 있음.
* 위의 예시에서는 0.0159로 0.05보다 작음. 따라서 유의미한 차이.
* 구체적으로 어떤 수준(집단)이 차이가 있는지 확인하려면 사후분석(post hoc tests)
* 유의미한 차이가 없는 경우에는 사후분석할 필요가 없음

**다원분산분석**

* 집단을 구분하는 변수(즉, 요인)이 두 개일 때 이원분산분석(two-way ANOVA)라 함
* 요인이 세 개이면, 삼원분산분석(three-way ANOVA)라 함
* 일반적인 표현으로, 요인이 n개 일 때, n원분산분석(n-way ANOVA)라고 함
* 다원분산분석을 실시하는 주요 목적 중 하나는 요인 간 상호작용을 파악하기 위함임


```python
dat = pd.read_csv('https://raw.githubusercontent.com/Liam427/stuydy-data/main/data/poison.csv', index_col=0)
dat.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>poison</th>
      <th>treat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.31</td>
      <td>1</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.45</td>
      <td>1</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.46</td>
      <td>1</td>
      <td>A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.43</td>
      <td>1</td>
      <td>A</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.36</td>
      <td>2</td>
      <td>A</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 이 데이터에서 종속변수는 time, 독립변수는 poison과 treat이다.
# poison 요인으로 구분한 집단별 표본수는 모두 16으로 동일

dat.groupby('poison').agg(len)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>treat</th>
    </tr>
    <tr>
      <th>poison</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>16.0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16.0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
# treat 요인구분한 집단별 표본수는 모두 12으로 동일


dat.groupby('treat').agg(len)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>poison</th>
    </tr>
    <tr>
      <th>treat</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>12.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>B</th>
      <td>12.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>C</th>
      <td>12.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>D</th>
      <td>12.0</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
# poison과 treat 요인으로 구분한 각 집단별 표본수는 모두 4로 동일

dat.groupby(['poison', 'treat']).agg(len)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>time</th>
    </tr>
    <tr>
      <th>poison</th>
      <th>treat</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">1</th>
      <th>A</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>B</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>C</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>D</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">2</th>
      <th>A</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>B</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>C</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>D</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">3</th>
      <th>A</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>B</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>C</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>D</th>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
model = ols('time ~ C(poison) * C(treat)', dat).fit()
anova_lm(model)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>df</th>
      <th>sum_sq</th>
      <th>mean_sq</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C(poison)</th>
      <td>2.0</td>
      <td>1.033012</td>
      <td>0.516506</td>
      <td>23.221737</td>
      <td>3.331440e-07</td>
    </tr>
    <tr>
      <th>C(treat)</th>
      <td>3.0</td>
      <td>0.921206</td>
      <td>0.307069</td>
      <td>13.805582</td>
      <td>3.777331e-06</td>
    </tr>
    <tr>
      <th>C(poison):C(treat)</th>
      <td>6.0</td>
      <td>0.250138</td>
      <td>0.041690</td>
      <td>1.874333</td>
      <td>1.122506e-01</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>36.0</td>
      <td>0.800725</td>
      <td>0.022242</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



* poison: F(2, 36) = 23.222, p < 0.05로 유의미. 즉 poison의 수준에 따라 평균에 차이가 난다고 볼 수 있음
* treat: F(3, 36) = 13.806, p < 0.05로 유의미. 즉 treat의 수준에 따라 평균에 차이가 난다고 볼 수 있음
* poison:treat: F(6, 36) = 1.874, p > 0.05로 유의미하지 않음. 상호작용 효과는 발견하지 못함

#### **4) One-sample Chi-square test**

* 적합성검정  
    $H_0$ : 관찰빈도 = 기대빈도  
    $H_1$ : 관찰빈도 ≠ 기대빈도


```python
# 관찰빈도 
xo = [324, 78, 261] 

# 기대빈도 
xe = [371, 80, 212] 

dfx = pd.DataFrame([xo, xe], 
                   columns = ['A','B','C'], 
                   index = ['Obs', 'Exp'])
dfx
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Obs</th>
      <td>324</td>
      <td>78</td>
      <td>261</td>
    </tr>
    <tr>
      <th>Exp</th>
      <td>371</td>
      <td>80</td>
      <td>212</td>
    </tr>
  </tbody>
</table>
</div>




```python
# bar plot

ax = dfx.plot(kind='bar', figsize=(8,6))
ax.set_ylabel('value')
plt.grid(color='darkgray')
plt.show();
```


    
<p align="center">
  <img src="/images/2021-01-27-AIbootcampReview2_files/2021-01-27-AIbootcampReview2_27_0.png" alt="Review2"/>
</p>
    



```python
# 카이제곱 통계량  

from scipy.stats import chisquare

chiresult = chisquare(xo, f_exp=xe) 
chiresult
```




    Power_divergenceResult(statistic=17.329649595687332, pvalue=0.00017254977751013492)



* p-value가 0.0001725로 유의수준 0.05보다 아주 작으므로 귀무가설을 기각하고, 대립가설을 지지한다.

#### **5) Two-sample Chi-square tes**


```python
xf = [269, 83, 215]
xm = [155, 57, 181]

xdf = pd.DataFrame([xf, xm], 
                   columns = ['item1','item2','item3'],
                   index = ['Female','Male'])
xdf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item1</th>
      <th>item2</th>
      <th>item3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female</th>
      <td>269</td>
      <td>83</td>
      <td>215</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>155</td>
      <td>57</td>
      <td>181</td>
    </tr>
  </tbody>
</table>
</div>



* 독립성 검정  
    $H_0$: 성별과 아이템 품목 판매량은 관계가 있다.  
    $H_1$: 성별과 아이템 품목 판매량은 관계가 없다.  


```python
from scipy.stats import chi2_contingency 

chi_2, p, dof, expected = chi2_contingency([xf, xm]) 
msg = 'Test Statistic: {}\np-value: {}\nDegree of Freedom: {}' 
print(msg.format(chi_2, p, dof)) 
print(expected)
```

    Test Statistic: 7.094264414804222
    p-value: 0.028807134195296135
    Degree of Freedom: 2
    [[250.425   82.6875 233.8875]
     [173.575   57.3125 162.1125]]
    

* p-value는 0.02881으로 유의수준 0.05보다 작은 값이므로 귀무가설을 기각한다.
* 따라서 성별과 아이템 품목 판매량은 관계가 없다.

#### 6) Bayesian


```python

```


<p align="center">
    <a href="https://codestates.com" target = "_blank">
        <img src="https://i.imgur.com/RDAD11M.png" 
        width="300" height="300"
        alt="codestates"/>
    </a>
</p> 
