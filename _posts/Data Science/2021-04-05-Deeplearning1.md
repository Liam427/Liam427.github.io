---
title: "Optimization, SGD, Momentum, AdaGrad, Adam"
layout: single
use_math: true
author_profile: true
read_time: true
comments: true
share: true
related: true
categories:
    - AI Bootcamp
    - Deep Learning
tag:
    - AI Bootcamp
    - Deep Learning
    - Optimization
    - 최적화
    - 매개변수 갱신
    - 확률적 경사 하강법
    - SGD
    - 모멘텀
    - AdaGrad
    - Adam
    - Diary
    - Post
    - Codestates
---
<p align="center">
  <img src="/assets/img/post/AIbootcamp.jpg" alt="AI Bootcamp"/>
</p>  

# Deep Learning 첫번째  
---  

**신경망(딥러닝) 학습의 효율과 정확도를 높일 수 있는 여러가지 기법들을 알아보겠습니다.**  

### 1. 매개변수 갱신  
신경망 학습의 목적은 손실 함수의 값을 가능한 한 낮추는 매개변수를 찾는 것입니다. 이는 곧 매개변수의 최적값을 찾는 문제이며, 이러한 문제를 푸는것을 **최적화**(optimization)라 합니다.  
  
  
### 2. 확률적 경사 하강법(SGD)  
$W \leftarrow W-\eta\frac{\partial L} {\partial W}$  
의 수식으로 나타낼수 있습니다. 여기에서 $W$ 는 갱신할 가중치 매개변수고 $\frac{\partial L} {\partial W}$은 $W$ 에 대한 손실 함수의 기울기입니다. $\eta$ 는 학습률을 의미하는데, 실제로는 0.01dlsk 0.001과 같은 값을 미리 정해서 사용합니다. 여기에서 SGD는 특정 함수에 따라 굉장히 비효율적인 음직임을 보일수도 있습니다. 즉, 비등방성(anisotropy)함수(방향에 따라 성질, 즉 여기에서는 기울기가 달라지는 함수) 에서는 탐색 경로가 비효율적이라는 것입니다.  


### 3. 모멘텀  
모멘텀(momentum)은 운동량을 뜻하는 단어로, 물리와 관계가 있습니다. 모멘텀 기법은 수식으로는 다음과 같이 쓸 수 있습니다.  
$v \leftarrow \alpha v- \eta\frac{\partial L} {\partial W}$  
$W \leftarrow W+v$   
$W$ 는 갱신할 가중치 매개변수, $\frac{\partial L} {\partial W}$ 는 $W$ 에 대한 손실 함수의 기울기, $\eta$ 는 학습률입니다. $v$ 는 물리에서 말하는 속도(velocity)에 해당합니다.  
SGD와 비교하면 지그재그의 정도가 덜한 것입니다. 이는 $x$ 축의 힘은 아주 작지만 방향은 변하지 않아서 한 방향으로 일정하게 가속하기 때문입니다. 거꾸로 $y$ 축의 힘은 크지만 위아래로 번갈아 받아서 상충하여 $y$ 축 방향의 속도는 안정적이지 않습니다. 전체적으로 SGDqhek $x$ 축 방향으로 빠르게 다가가 지그재그 움직임이 줄어듭니다.  

### 4. AdaGrad  
신경망 학습에서는 학습률($\eta$)값이 중요합니다. 이 값이 너무 작으면 학습시간이 너무 길어지고, 반대로 너무 크면 발산하여 학습이 제대로 이뤄지지 않습니다. 이 학습률을 정하는 효과적 기술로 **학습률 감소**(learning rate decay)가 있습니다. 이는 학습을 진행하면서 학습률을 점차 줄여가는 방법입니다. 처음에는 크게 학습하다가 조금씩 작게 학습한다는 얘기로, 실제 신경망 학습에 자주 쓰입니다. 학습률을 서서히 낮추는 가장 간단한 방법은 매개변수 전체의 학습률 값을 일괄적으로 낮추는 것이겠죠. 이를 더욱 발전시킨 것이 AdaGrad입니다. AdaGrad는 각각의 매개변수에 맞춤형값을 만들어 줍니다. AdaGrad의 갱신방법은 수식으로 다음과 같습니다.  
$h \leftarrow h+\frac{\partial L} {\partial W}\odot \frac{\partial L} {\partial W}$  
$W \leftarrow W-\eta\frac{1} {\sqrt h}\frac{\partial L} {\partial W}$  
$W$ 는 갱신할 가중치 매개변수, $\frac{\partial L} {\partial W}$ 은 $W$ 에 대한 손실 함수의 기울기, $\eta$ 는 학습률을 뜻합니다. $h$ 는 기존 기울기값을 제곱하여 계속 더해주게 되고($\odot$ 은 행렬의 원소별 곱을 의미합니다.), 매개변수를 갱신할 때 $\frac{1} {\sqrt h}$ 을 곱해 학습률을 조정합니다. 매개변수의 원소 중에서 많이 움직인 원소는 학습률이 낮아진다는 뜻인데, 다시 말해 학습률 감소가 매개변수의 원소마다 다르게 적용됨을 뜻합니다. 이런 수식적용을 통해 $y$ 축 방향의 기울기는 처음에 크게 움직이지만, 큰 움직임에 비례해 갱신 정도도 큰 폭으로 작아지도록 조정됩니다. 그래서 $y$ 축 방향으로 갱신 강도가 빠르게 약해지고, 지그재그 움직임이 줄어듭니다.  

### 5. Adam  
모멘텀은 공이 그릇 바닥을 구르는 듯한 움직임을 보였습니다. AdaGrad는 매개변수의 원소마다 적응적으로 갱신 정도를 조정했습니다. 이 두 기법을 융합한 기법이 바로 **Adam**입니다.  
이 두 방법의 이점을 조합했기 때문에 매개변수 공간을 효율적으로 탐색을 할 수 도있고, 하이퍼파라미터의 편향 보정이 진행된다는 점도 Adam의 특징입니다. Adam 갱신과정은 그릇 바닥을 구르듯 움직입니다. 모멘텀과 비슷한 패턴인데, 모멘텀 보다 공의 좌우 흔들림이 적습니다. 이는 학습의 갱신 강도를 적응적으로 조정해서 얻는 혜택입니다.  

**SGD, 모멘텀, AdaGrad, Adam의 네 후보 중 어느것이 좋을까요? 아직 모든 모델에서 완벽한 기법은 없습니다. 지금도 각각의 기법에 대한 연구가 활발히 진행되는 것으로 알고있습니다. 모델에 맞춰서 고민해 가며 사용해야 할 것입니다.**
