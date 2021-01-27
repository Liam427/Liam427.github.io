---
title: " AI Bootcamp Section1 Review"
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

# # AI Bootcamp Section1 Review  
---  
내가 section1을 학습하며 배운 내용을 토대로 백지복습을 하려 한다.  
알고 있는 내용들은 이미 블로그에 포스팅을 해 놓았고, github에 fork했으니,  
내가 모르고 있었거나 좀 더 공부 하고 싶은 내용들을 적어 보겠다.  
아쉬운 점은 아직 jekyll blog와 github등 어려움이 있어 plotly를 올리지 못했고, 이미지로 대체 하겠다.  

## 1. Exploratory Data Analysis  

* week1에서는 다른 부분들은 괜찮았으나 plotly라는 라이브러리를 처음 보았고, 응용해보고 싶다는 생각이 무척이나 들었다.  
    그 이유중에 하나는 나는 앞으로 금융권이나, 핀테크쪽에서 데이터를 만져보고 싶다고 부끄럽지만 생각을 하고 있기때문에  
    plotly가 많이 매력적이게 다가 왔다.  

```python
# 일단 plotly를 설치 하겠다. chart_studio를 설치하면 plotly가 설치 된다.  
# cufflinks를 설치해주면 pandas에서 바로 plotly로 넘겨 시각화 해준다.  

pip install chart_studio  
pip install cufflinks  
```  

```python  
# plotly import

import chart_studio.plotly as py
import cufflinks as cf
cf.go_offline(connected=True)  
```  
### 예제  
```python
# Cufflinks in Python
# https://plot.ly/ipython-notebooks/cufflinks/

df = cf.datagen.lines()
df.head()  
```  

||OJY.BD|TER.II|EZV.WN|KON.AW|QRY.FL|
|:-:|:-:|:-:|:-:|:-:|:-:|
|2015-01-01|1.550663|1.396062|-0.217668|0.735333|-0.101336|
|2015-01-02|1.161382|0.268030|-0.716412|3.002882|-0.958505|
|2015-01-03|0.691182|1.705918|-0.278059|2.316131|-0.863929|
|2015-01-04|0.708755|1.710140|-0.292352|3.297931|-0.374006|
|2015-01-05|2.662028|1.379902|-0.242189|2.020648|-0.503709|  

**1) line plot**  

```python
df.iplot(kind='line')
```  
<p align="center">
  <img src="/images/2021-01-27-AIbootcampReview1/plotly1.PNG" alt="plotly"/>
</p>  

**2) bar plot**  
```python
df.iplot(kind='bar')
```  
<p align="center">
  <img src="/images/2021-01-27-AIbootcampReview1/plotly2.PNG" alt="plotly"/>
</p>  

```python
df.iplot(kind='bar', barmode='stack')
```  
<p align="center">
  <img src="/images/2021-01-27-AIbootcampReview1/plotly3.PNG" alt="plotly"/>
</p>  

```python
df['OJY.BD'].iplot(kind='bar')
```  
<p align="center">
  <img src="/images/2021-01-27-AIbootcampReview1/plotly4.PNG" alt="plotly"/>
</p>  

```python
df.iplot(kind='barh', barmode='stack')
```  
<p align="center">
  <img src="/images/2021-01-27-AIbootcampReview1/plotly5.PNG" alt="plotly"/>
</p>  

**3) area chart**  
```python
df.iplot(kind='area')
```  
<p align="center">
  <img src="/images/2021-01-27-AIbootcampReview1/plotly6.PNG" alt="plotly"/>
</p>  

```python
df.iplot(kind='area', fill=True)
```  
<p align="center">
  <img src="/images/2021-01-27-AIbootcampReview1/plotly7.PNG" alt="plotly"/>
</p>  


* 대략 plot의 종류는 아래에 나열한 정도가 있다.  
    scatter, bar, box, spread, ratiom, heatmap, surface, histogram, bubble, bubble3d, scatter3d, scattergeo, ohlc, candle, pie, horoplet   
* plotly 세부 설정이 있지만, 포스팅이 너무 길어질것 같아 다음에 자세하게 다루겠다.

