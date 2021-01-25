---
title: "Jekyll blog에 MathJax로 수학식 표현하기"
layout: single
use_math: true
author_profile: true
read_time: true
comments: true
share: true
related: true
categories:
    - Develop
tag:
    - Post
    - Mathjax
    - Python
---  

[MathJax](https://github.com/mathjax/MathJax)를 사용하면, 수학식을 표현할 수 있어요.  

### Jekyll에 MathJax 적용하기  

#### 1. mathjax_support.html 파일 생성하기  

`_includes/mathjax_support.html` 파일을 생성한 후 아래 코드를 입력합니다.  

```
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    TeX: {
      equationNumbers: {
        autoNumber: "AMS"
      }
    },
    tex2jax: {
    inlineMath: [ ['$', '$'] ],
    displayMath: [ ['$$', '$$'] ],
    processEscapes: true,
  }
});
MathJax.Hub.Register.MessageHook("Math Processing Error",function (message) {
      alert("Math Processing Error: "+message[1]);
    });
MathJax.Hub.Register.MessageHook("TeX Jax - parse error",function (message) {
      alert("Math Processing Error: "+message[1]);
    });
</script>
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
```  
#### 2. layouts 에 추가하기  

`_layouts/default.html` 파일을 열어 `<head>` 태그 아래 부분에 아래 코드를 추가합니다.  

```
{% if page.use_math %}
  {% include mathjax_support.html %}
{% endif %}
```  
#### 3. MathJax 사용하기  

마크다운 파일에 YAML Front Matter부분에 `use_math: true` 를 추가합니다.  

```
---
title: "MathJax로 수학식 표현하기"
use_math: true
tags:
  - MathJax
  - Jekyll
---
```  

#### 4. MathJax 표현한 예제  

```
In N-dimensional simplex noise, the squared kernel summation radius $r^2$ is $\frac 1 2$
for all values of N. This is because the edge length of the N-simplex $s = \sqrt {\frac {N} {N + 1}}$
divides out of the N-simplex height $h = s \sqrt {\frac {N + 1} {2N}}$.
The kerel summation radius $r$ is equal to the N-simplex height $h$.
```  

In N-dimensional simplex noise, the squared kernel summation radius $r^2$ is $\frac 1 2$
for all values of N. This is because the edge length of the N-simplex $s = \sqrt {\frac {N} {N + 1}}$
divides out of the N-simplex height $h = s \sqrt {\frac {N + 1} {2N}}$.
The kerel summation radius $r$ is equal to the N-simplex height $h$.  

#### 5. References  

[MathJax 문법](https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:TeX_%EB%AC%B8%EB%B2%95)