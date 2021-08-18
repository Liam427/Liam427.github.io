---
title: "구글 colab run time 끊김 방지!"
layout: single
author_profile: true
read_time: true
comments: true
share: true
related: true
categories:
    - Develop
tag:
    - Colab
    - Post
    - Python
---  
<p align="center">
  <img src="/assets/img/post/colab.png" alt="colab"/>
</p> 

### Colab run time 끊김 방지!!!
colab을 사용하다 보면 정확한 시간은 모르지만 대략적으로 1시간 이상 아무런 동작이 없으면
run time이 끊기게 되어 3시간에 걸쳐 돌려놓은 model 결과가 날라가고.... 날라가고...
현상이 있게 된다. 그것을 방지 하기 위해 아래의 방법을 적용하자!!!

```python
# Colab창에서 F12 → console → 맨마지막 줄에 입력하고 숫자가 뜨면 OK
function ClickConnect(){
console.log("Working"); 
document
  .querySelector('#top-toolbar > colab-connect-button')
  .shadowRoot.querySelector('#connect')
  .click() 
}
setInterval(ClickConnect,60000)
```