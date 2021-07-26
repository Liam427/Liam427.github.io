---
title: "Porfolio With Python 1"
layout: single
use_math: true
author_profile: true
read_time: true
comments: true
share: true
related: true
categories:
    - Finance
tag:
    - Post
    - Finance
    - Book Review
---  
<p align="center">
  <img src="/assets/img/post/portfoliowithpython.jpg" alt="Finance"/>
</p> 
# 1. NPV와 IRR

투자 세계에서 순현재가치(Net Present Value, NPV)와 내부수익률(Internal Rate of Return, IRR)은 그들만의 언어다.  
NPV나 IRR만 보고 투자 여부를 결정하는 것은 아니지만, 사업을 계획하거나 투자 제안을 받는 경우 가장 먼저 묻는 것이 NPV와 IRR이다.

## 1.1 NPV

* NPV는 현재의 액면가를 나타내는 현금흐름의 순가치다.  
* NPV가 0 이상이면 수익성이 있다고 판단해 투자안을 채택하고, 그렇지 않다면 투자안을 포기한다.  

<center>$NPV = SUM(\frac{현금흐름}{(1+할인율)^i})$</center>    
<center>$NPV = \sum \frac{C}{(1+rate)^i}$</center>  
<center>C = 현금흐름, rate = 할인율, i = 기간</center>


```python
# 현금흐름을 cashflows 리스트에 저장한다.
# i는 햇수, r은 이자율이다.
cashflows_func = [30000, 33000, 37000, 40000, 45000]
i = 0
r = 0.015

# 최초 투자금액이며 현금 유출이므로 (-)로 표시
npv_func = -90000
```


```python
# cashflows 리스트를 반복해 미래에 들어올 현금흐름을 할인함으로써 현재가치로 계산한 다음 npv 변수에 누적
for c in cashflows_func:
    i = i + 1
    npv_func = npv_func + c / (1 + r) ** i

print(npv_func)
```

    86431.30057480175
    


```python
# numpy 라이브러리는 순현재가치를 계산하는 npv 함수를 제공한다
import numpy_financial as npf

# 현금흐름을 cashflows 리스트에 저장한다.
cashflows_func = [30000, 33000, 37000, 40000, 45000]
r = 0.015
```


```python
# npv 함수로 순현재 가치를 계산한다.
npv_np = npf.npv(r, cashflows_func)
print(npv_np)
```

    179077.77008342376
    

* npv_func과 npv_np의 값이 다른이유는 npv_func는 i=0부터 시작하고, npv_np는 i=1부터 시작하기 때문에 다르게 나타난다.  
    그렇기 때문에 **chshflows** list를 다르게 변형을 해줘야 한다.


```python
# chshflows의 최초 투자금액을 제일앞에 입력해준다.
cashflows_np = [-90000, 30000, 33000, 37000, 40000, 45000]
npv_np_modify = npf.npv(r, cashflows_np)
print(npv_np_modify)
```

    86431.30057480175
    

* 이렇게 cashflows list에 NPV의 공식대로 최초 투자금액을 입력해주니 함수로 만들어낸 npv와 numpy를 이용한 npv가 같아졌다.

## 1.2 IRR

* NPV가 0인 경우의 할인율이 IRR이다.  
* IRR > 기업의 자본비용(시장이자율)일 경우 그 사업을 채택하고, 그렇지 않으면 사업을 기각한다.  
* IRR은 투자의 수익성을 측정하는 지표이지만, 이것만 갖고 의사결정을 하지는 않는다. 투자 기간이 짧고 초반 현금 유입이 클수록 IRR은 높을 수 있다. 그러나 장기적으로 현금 유입이 더 많은 경우에는 앞서 언급한 단기 투자안보다 가치가 높을 수 있기 때문이다.  
<center>$IRR = NPV = \sum \frac{C_t}{(1+rate)^t}-C_0$</center>


```python
# 현금흐름을 cashflows_irr list에 저장한다.
# 이도 위와 같이 numpy library를 이용할 것이기 때문에 list에 최초 현금 투자금액을 입력해준다.
cashflows_irr = [-90000, 30000, 33000, 37000, 40000, 45000]

# numpy library의 IRR함수를 사용해 내부수익률을 계산한다.
irr = npf.irr(cashflows_irr)

#구한 IRR을 NPV의 할인율로 사용해 NPV를 구한다. IRR이 정확하다면 NPV는 0이다.
npv = npf.npv(irr, cashflows_irr)
```


```python
print('IRR {0:.1%} makes NPV {1:.2f}'.format(irr, -npv))
```

    IRR 27.6% makes NPV 0.00
    

* 위의 결과를 보건데 IRR = 27.6%가 나왔으므로 기업의 자본비용보다 높아 사업을 채택 할 것이고, NPV가 0이므로 IRR은 정확한 값을 가진다.

# 2. 수익률 대 수익률

## 2.1 수익률과 할인율의 개념

* 수익률이란 미래의 시점에 발생하는 모든 현금(현금흐름)합계액과 현재의 가치를 일치시키는 할인율을 말한다. 즉, 현재의 시점에서 미래의 가치를 계산하는 데 사용되면 수익률이지만, 역으로 미래의 가치를 현재의 시점으로 되돌려 계산한다면 할인율이 된다.  
* 시장에서 말하는 수익률은 만기수익률을 가리키는데 시장수익률, 최종수익률, 내부수익률과 같은 개념으로 사용된다.

## 2.2 기간 수익률의 평균, 산술평균과 기하평균

### 2.2.1 산술평균

* 평균은 여러 개의 자료를 대표하는 역할을 하며, 보통 평균이라 하면 '산술평균(arithmetic mean)'을 생각한다.
* 산술평균은 회사 내 급여 수준, 학급별 시험 점수, 계절별 기온 등 광범위하게 사용된다.
<center>$산술평균 = \frac{1}{n}(r_1+r_2+r_3\cdots+r_{n-1}+r_n)$</center>


```python
# 기간별 수익률을 returns list에 담는다.
returns = [0.1, 0.07, 0.04]

# 합계를 저장할 변수를 준비한다.
SumOfReturn = 0.0

# 평균을 저장할 변수를 준비한다.
arimean = 0.0

# 기간별 수익률의 데이터 개수를 구한다.
n = len(returns)
```


```python
# returns list를 for 루프로 반복한다. 반복하는 동안 각 수익률을 변수 r로 받는다.
for r in returns:
    SumOfReturn = SumOfReturn + r
    
arimean = SumOfReturn / n
print('AriMean is {:.2%}'.format(arimean))
```

    AriMean is 7.00%
    

### 2.2.2 기하평균

* 기하평균은 수치의 곱을 수치의 개수로 제곱근을 취해 구한다.
* 기하평균은 물가상승률, 매출증가율, 인구성장률, 투자이율 등 성장률의 평균을 산출할 때 사용한다.
    * 대표적인 예가 CAGR(Compound Average Growth Rate)로 국가의 GDP나 기업 매출액의 성장률을 나타내는 수치이다.  
<center>$기하평균 = \sqrt[n]{(1+r_1)(1+r_2)(1+r_3)\cdots(1+r_{n-1})(1+r_n)} -1$</center>


```python
# 평균을 저장할 변수를 준비한다.
geomean = 1.0

# return list를 for 루프로 반복한다. 반복하는 동안 각 수익률을 변수 r로 받는다.
for r in returns:
    geomean = geomean * (1 + r)

# 기간 수익률로 변환한다.
geomean = geomean ** (1 / n) - 1

# 기하평균을 출력한다. 문자열의 포맷(format)을 이용해 출력 양식을 만든다. {}는 geomean 변수의 출력 위치인데, 
# 그 안의 :.2%는 소수점 둘째 자리(.2)로 백분율 %을 표현하라는 의미다.
print('GeoMean is {:.2%}'.format(geomean))
```

    GeoMean is 6.97%
    

## 2.3 지배원리

* 금융시장의 여러 자산(주식, 채권, 파생상품 등)을 조합하면 무수히 많은 포트폴리오가 나올 것이다. 이 포트폴리오를 '기대수익률'과 '위험'이라는 기준으로 재단 할 수 있다.  
* 포트폴리오부터 얻는 기대수익률은 클수록 좋고, 표준편차 또는 분산으로 표현하는 위험은 작을수록 좋다.  
    * 위험회피형 투자자는 두 포트폴리오 기대수익률이 동일하다면 표준편차가 작은 포트폴리오를 선택할 것이다. 즉, 두 포트폴리오 수익률의 표준편차가 동일하다면 기대수익률이 상대적으로 큰 포트폴리오를 선택할 것이다. 이를 평균-분산 기준(mean-variance criterion) 또는 지배원리(dominance principle)라고 한다.

* Example  

<center>

|투자대상|기대수익률(%)|표준편차(%)|
|:---:|:---:|:---:|
|A|7.0|2.0|
|B|7.0|3.0|
|C|9.0|2.0|
|D|9.0|3.0|  

</center>

* 투자자가 합리적일 경우 위험이 동일 하다면 기대수익률이 가장 높은 포트폴리오를 선택할 것이며, 시대수익률이 동일하다면 위험이 가장 낮은 포트폴리오를 선택할 것이다.
* 포트폴리오 A와 B는 기대수익률이 같지만 포트폴리오 A의 표준편차가 낮기 때문에 포트폴리오 B를 지배하게 된다.
* 포트폴리오 C는 A와 표준편차는 같지만 투자자는 기대수익률이 높은 포트폴리오 C를 선택하게 된다.  
* 포트폴리도 D는 포트폴리오 C와 기대수익률은 같지만 표준편차가 높다. 따라서 지배원리에 따라 포트폴리오 C에 밀려 선택받지 못할 것이다.  
* 평균-분산 지배원리의 오류라는 것이 있다. 포트폴리오 A와 D는 지배원리만으로 선택할 수 없다. 포트폴리오 C는 A에 비해 기대수익률이 높지만, 위험을 가리키는 표준편차도 높다. 반대로 포트폴리오 A는 C에 비해 위험을 가리키는 표준편차는 낮지만 기대수익률 역시 낮다. 즉, 지배원리만 갖고 해결할 수 없는 것이다.
