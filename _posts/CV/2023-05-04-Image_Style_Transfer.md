---

title:  "[Paper Review] Image Style Transfer Using Convolutional Neural Networks"

excerpt: "Image Style Transfer 논문 리뷰"
categories:
  - CV
toc_label: "  목차"
toc: true
toc_sticky: true

date: 2023-05-04
last_modified_at: 2023-05-04
---

## Background

Texture tranfer란 한 이미지의 style을 다른 image로 전달하는 task를 의미한다. 초창기 texture tranfer는 non-paramatic algorithm을 통하여 texture 합성을 진행할 수 있었지만 여기에는 low-level feature만을 사용한다는 한계점이 존재한다. 일반적으로 원본 이미지의 style로부터 content를 분리시키는 일은 매우 어려운 문제이다. 그러나 이상적인 style tranfer algorithm은 target으로부터 중요한 이미지 content를 추출하고 source image의 스타일을 정제하여 texture tranfer procedure에 전달할 수 있어야한다.  

Deep Convolution Neural Network는 이미지의 high-level semantic information을 추출할 수 있기 때문에 style transfer가 가능하며, 논문에서 소개되는 모델은 CNN을 활용함으로써 style과 content를 독립적으로 처리하고 조작하여 원본 이미지의 style로부터 content를 분리하게 된다.

## Deep image representations

<p style="text-align: center;">
  <img src="/images/style_transfer_1.png" width="50%">
</p>

### Content representation

Content image에 대한 학습을 위하여 원본 이미지와 생성된 이미지 간의 MSE Loss를 활용한다. 아래의 식은 이 Loss term과 미분식을 나타낸다. 


$$
L_{content}(\vec{p},\vec{x}, l)= {1\over2} \sum_{i,j}(F_{ij}^l-P_{ij}^l)^2.
$$

$$
{\partial L_{content}\over \partial F_{ij}^l}= \begin{cases}
	   (F^l-P^l)_{ij} & \text{if $F^l_{ij}$ > 0}\\
            0 & \text{if $F^l_{ij}$ < 0}
		 \end{cases}
$$

이 때 $F_{i,j}^l$은 생성된 이미지의 layer $l$에서 j번째 위치에 존재하는 i번째 filter의 activation을, $P_{i,j}^l$은 content original image의 layer $l$에서 j번째 위치에 존재하는 i번째 filter의 activation을 의미한다. $\vec{p},\vec{x}$는 각각 content original image와 생성된 image를 의미한다. 

이때 network의 higher layer (Figure 1. d와 e)는 물체나 정렬에 대한 high-level content를 포착하고 lower layer (Figure 1. a-c)는 단순히 원본 이미지의 정확한 픽셀 값을 재생산하는데 집중한다. 여기서 저자들은 higher layer의 feature map을 content representation을 위한 방식으로 활용한다. 

공개된 코드 상에서 저자들은 16개의 CNN과 5개의 Pooling layer로 구성된 VGG16에서 normalized된 version의 feature map을 활용하는데, Figure1의 d정도 위치에 존재하는 3번째 pooling layer 이후의 4_2 conv layer를 content layer의 feature map으로 사용한다.


### Style representation

Input image의 style 표현을 추출하기 위해서는 feature space의 정보를 활용할 필요가 있다. style 정보는 content와는 다르게 직접적인 layer의 feature을 뽑는 것이 아닌 feature간의 상관관계를 계산하여 구한다. 이를 위하여 Gram matrix $G^l$이 이 도입된다. $G_{i,j}^l$은 layer $l$에서 vector화된 feature map $i$와 $j$간 내적을 한 것을 의미한다. 저자들은 원본 이미지의 Gram matrix와 생성 이미지의 Gram matrix간의 mean-squared distance를 최소화하는 방향으로 Loss를 구성한다. $A^l$을 style original image에서 layer $l$에서 layer $l$에 위치하는 feature map이라고 하고, $F^l$을 생성 이미지에서 layer $l$에서 layer $l$에 위치하는 feature map이라고 했을 때 아래와 같이 Loss를 표현할 수 있다.

$$
L_{style}(\vec{a}, \vec{x})=\sum_{l=0}^L w_lE_l, \ E_l={1\over4N_l^2M_l^2} \sum_{i,j} (G^l_{i,j}-A^l_{i,j})^2.
$$

이 때 $w_l$은 style loss를 위한 각 layer 별 weighting factor을 의미한다. 위 식을 미분하면 아래와 같은 식을 얻을 수 있다. 

$$
{\partial E_l\over \partial F_{ij}^l}= \begin{cases}
	   {1\over{N_l^2M_l^2}}((F^l)^T(G^l-A^l))_{ij} & \text{if $F^l_{ij}$ > 0}\\
            0 & \text{if $F^l_{ij}$ < 0}
		 \end{cases}
$$

### Style tranfer

최종적으로 얻기를 원하는 것은 content 정보를 가진 $\vec{p}$와 style 정보를 가진 $\vec{a}$를 input image $\vec{x}$에 합성하는 것이다. 이를 위한 최종적인 Loss funtion은 아래와 같다. 

$$
L_{total}(\vec{p},\vec{a},\vec{x})=\alpha L_{content}(\vec{p}, \vec{x}) +\beta L_{style}(\vec{a}, \vec{x})
$$

이때, $\alpha$와 $\beta$는 weight factor이며, 위의 Loss를 통하여 random white noise의 pixel 값들이 업데이트 된다. 

## Results

Figure2는 Style Transfer 알고리즘을 도식화한 것이다.

<p style="text-align: center;">
  <img src="/images/style_transfer_2.png" width="80%">
</p>

Content image와 Style image가 존재하고, 생성할 이미지 $\vec{x}$는 white noise로부터 시작해서 content 정보와 style 정보를 합성하여 얻어낸다.

### Trade-off between content and style matching

<p style="text-align: center;">
  <img src="/images/style_transfer_3.png" width="50%">
</p>

Figure3은 content와 style image의 weighting factor의 비율을 변화시키며 실험한 것이다. 결과적으로 style 정보를 강하게 넣고 싶다면 $\beta$의 값을 높이거나 $\alpha$의 값을 낮추면 된다. 반대로 content 정보를 강하게 넣고 싶다면 $\beta$의 값을 낯추거나 $\alpha$의 값을 높이면 된다.

### Effect of different layers of the Convolutional Neural Network

실험 결과 style tranfer를 위해서는 더 구체적인 pixel 정보를 함유한 lower layer보다 higher layer을 사용하는 것이 좋다.

### Initialization of gradient descent

실험 결과 white noise로 시작한 이미지는 모두 다양한 새로운 이미지를 생성한 것에 반해 content image나 style image로 초기화해서 진행했을 경우 모두 같은 결과를 갖게 된다. 따라서 다양한 새로운 이미지를 보고싶다면 white noise로 시작하는 것이 좋다.

## Discussion

본 알고리즘이 가진 기술적 결함은 먼저 해상도 문제가 있다. 본 논문에서는 512x512의 이미지 해상도를 가지고 한시간을 걸려 결과를 얻어낸다. 즉, 이미지의 해상도가 커질수록 합성 결과를 나타내기까지 오랜 시간이 걸리게 된다. 

다음으로 대다수의 작업이 확실히 style과 content image를 분리해냈는지 또 그것을 설명할 수 있는지가 불분명하다. 이를 더 잘 설명할 수 있는 방법이 필요하다.





