---
title: " AI Bootcamp 세번째"
layout: single
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
  
# 세번째 Diary
---  

### 1. Flow  

> 1. 슬라이싱으로 원하는 부분만을 잘라서 가져온다.
> 2. concat & merge 함수로 데이터 합치기
> 3. tidy Data  ⇄ wide Data
> 4. groupby로 데이터를 구분지어 수학적 데이터 얻기  

### 2. Result & Info  

> 1. concat → Data 붙이는 연습  
    - `pd.concat()`
    - data를 붙인다.
    - axis = 0 → row로 붙인다. axis = 1 → col로 붙인다.
    - index 또는 col의 갯수가 달라서 빈값이면 NaN값으로 대체된다.
    - feature name = column name  
> 2. merge
    - `pd.merge()`
    - 공통된 부분을 기준으로 data를 붙인다.
> 3. conditioning
    - 하나의 변수를 지정하여 조건을 정해주는것.
    - `condition = (df[ ] 조건)` → 컨디션을 변수로 설정
    - dataframe에서 컨디션을 적용하여 선택
> 4. 측정수준
    - 분류(categorical data) = 명목척도, 순서척도
    - 수량(numerical data) = 구간척도, 비율척도
> 5. tidy data → data를 표현하는 방법중 하나.
    - `melt()` : wide data → tidy data
    - `pivot_table()` : tidy data → wide data
    - 시각화나 다른 라이브러리를 잘 쓰기 위해 정제 하는것.
  
### 3. Etc  

> 1. 함수의 detail을 help 함수로 꼼꼼히 보기.
> 2. 시각화를 시작했는데 마찬가지로 함수의 detail을 확인해 가며 적용시켜보고 쫌 더 이쁜 시각화를 위해 노력해보자!  
  
<p align="center">
    <a href="https://codestates.com" target = "_blank">
        <img src="https://i.imgur.com/RDAD11M.png" 
        width="300" height="300"
        alt="codestates"/>
    </a>
</p>