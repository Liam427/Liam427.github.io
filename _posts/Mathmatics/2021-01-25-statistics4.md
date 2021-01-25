---
title: "첫번째 통계학 - 이항분포"
layout: single
use_math: true
author_profile: true
read_time: true
comments: true
share: true
related: true
categories:
    - Mathmatics
tag:
    - Post
    - Statistics
---  
<p align="center">
  <img src="/assets/img/post/mathmatics.jpg" alt="Mathmatics"/>
</p>  

# 연속확률변수의 확률분포함수
---  

**연속확률변수는 확률변수가 특정 구간의 모든 값을 다 가질 수 있기 때문에 가질 수 있는 값들을 일일이 지칭할 수 없는 확률변수를 의미한다.**
이는 앞서 설명한 ["첫번째 통계학 - 이산확률변수, 확률분포함수"](https://liam427.github.io/mathmatics/statistics2/)에서 확인할 수 있다.  
따라서 연속확률변수 $X$의 확률분포를 표현하는데 있어서 $X$가 가질 수 있는 모든 값($x$)을 나열하며 확률을 대응시키기는 어렵다.  
이 경우에는 확률변수 $X$가 가질 수 있는 특정 구간에서 확률이 어떻게 분포하는가를 나타낼 수 있는 함수를 이용하게 된다.  
즉, 연속확률변수 $X$의 확률분포는 확률의 밀도를 나타내는 확률밀도함수에 의해 결정된다.  

> **확률밀도함수(probability density functoin)**  
    다음을 만족하는 함수 $f(x)$를 연속확률변수 $X$의 확률밀도함수라 한다.
    Ⅰ. 모든 $x$에 대해 $f(x) \geq 0$  
    Ⅱ. $P(a\leq X \leq b)$ = $\int_{a}^{b} f(x)\, dx$  
    Ⅲ. $P(-\infty \leq X \leq \infty)$ = $\int_{-\infty}^{\infty} f(x)\, dx$ = 1  

> **연속확률변수의 기대값과 분산**  
    기대값 : $E(X) = \int xf(x)\, dx = \mu$  
    분산 : $Var(x) = \int (x-\mu)^2f(x)\, dx = \sigma^2$  

이들을 이산확률변수의 경우와 비교해 보면, 이산확률변수의 경우에는 특정한 값 $x$를 갖게 되는 확률을 각각 구할 수 있기 때문에 각각의 값들을 다 더한 것이고, 연속확률변수의 경우에는 구간에 대한 확률로만 표현할 수 있기 때문에 적분의 계산방법을 응용하는 것일 뿐 기대값과 분산에 대한 의상의 차이는 없음을 알 수 있다. 따라서 ["첫번째 통계학 - 이산확률변수, 확률분포함수"](https://liam427.github.io/mathmatics/statistics2/)의 기대값과 분산의 성질은 연속확률변수에 대해서도 여전히 성립된다.  

**다음 포스팅에서는 통계학의 모든 분야에서 가장 중요하게 생각되는 대표적인 연속확률분포인 정규분포에 대해서 알아 보려 한다.**