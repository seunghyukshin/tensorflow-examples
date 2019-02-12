# tensorflow-examples

## 1. 피파19 선수들의 능력치를 이용한 Linear regression 예제
### 가설
    Age(나이),Overall(통합 능력치),Potential(잠재 능력치)를 입력하여 Value(몸값)을 예측한다.

### 결과
    예측 할 수 없는 값

### 실패 요인
    가설이 잘 못 되었다.
    나이,오버롤,포텐셜 외에 변수를 추가해준다 하더라도 예측했던 값이 나오지 않을 가능성 높음
    Linear하다는 근거가 없이 MLR식에 넣었기 때문

### 개선 방향
- normalized하여 cost값을 줄인다.\
inverse_transform 
- 변수를 줄여 linear형태를 찾는다.\
→ x:Overall, y:Value\
→애초에 데이터셋이 좋지 않았다.
![figure_1](https://user-images.githubusercontent.com/31649100/52390948-51809b80-2ade-11e9-82f0-2c91a2f8c9f3.png)\
ovearll에 해당하는 value값들 중 **최댓값 혹은 평균값**을 택한다면 데이터를 다루기 쉬울 것으로 보임.

---
### 개선결과
    Value값에 log10을 취한 뒤 선형구조로 만드는 방법을 택함
![why](https://user-images.githubusercontent.com/31649100/52454746-4a19ca80-2b90-11e9-8508-d3fe8d4b21aa.png)

![22](https://user-images.githubusercontent.com/31649100/52399748-7128bb80-2b00-11e9-8698-2687bbfd0914.png)

    cost를 0.18대로 만들었지만 원하는 선모양대로 나오지 않음.

---
### RMSE(root mean squared error)를 사용(기존 cost식에 root만)
     오히려 cost 값은 커졌지만 원하는 선모양이 나왔다.

![aa](https://user-images.githubusercontent.com/31649100/52455008-784bda00-2b91-11e9-821f-e5c60d54f9dc.png)

![default](https://user-images.githubusercontent.com/31649100/52454823-b399d900-2b90-11e9-94a1-1151c75fa921.png)

> loss마다 아웃라이어에 취약한 경우가 있다.
> 
> loss 값 자체는 scale에 영향을 받는거라 절대적인 값 차이는 크게 의미가 없다.
---
### Regularization
    종종 선형의 모양이 overfitting 되는 경향이 있어 regularization 해줄 필요가 생김.
[출처] https://yujuwon.tistory.com/entry/TENSORFLOW-Regularization

![20190212_171228](https://user-images.githubusercontent.com/31649100/52621010-9004c480-2ee9-11e9-944e-5397ee20d375.png)

![20190212_171601](https://user-images.githubusercontent.com/31649100/52621140-e7a33000-2ee9-11e9-9059-883ef24b6ca3.png)

    rmse를 빼고 regularization을 취해주었다. lambda값은 0.2로 주었다.
    cost가 상당히 낮아졌다.

    overfitting 되는 문제가 아직 있음.
    18000여개의 data를 모두 training set으로 취해서 주황색으로 칠한 부분으로 overfitting.
    → data split 해줄것.

### Data split
[출처] https://yujuwon.tistory.com/entry/TENSORFLOW-Regularization




# 

## 2. 분꽃 데이터셋을 이용한 Sotfmax Classification
### 출처
 https://alphago.pe.kr/entry/4-TensorFlow%EC%99%80-%EB%86%80%EC%9E%90-Softmax-Classification

### Data Set
    X: SepalLength, SepalWidth, PetalLength, PetalWidth (4개 항목)
    Y: Species - setosa, versicolor, virginica (3종류)

    training set : 25, 25, 25 (50)
    validation set: 10, 10, 10 (30)
    test   set : 15, 15, 15 (45)

### 결과
![1](https://user-images.githubusercontent.com/31649100/52391818-93abdc00-2ae2-11e9-9ae4-e38969e4e66a.png)

    learning rate=0.5로 training 결과 0.029의 cost

![2](https://user-images.githubusercontent.com/31649100/52391784-78d96780-2ae2-11e9-8c77-2e08070f0dfa.png)

    validation set에서 4개의 data 분류 실패함

### 실패요인
SpealLength|SepalWidth|PetalLength|PetalWidth|Species
-|-|-|-|-
6|2.7|5.1|1.6|versicolor
7.2|3|5.8|1.6|virginica
7.9|3.8|6.4|2|virginica
6.3|2.8|5.1|1.5|virginica

    versicolor와 virginica 분류에 실패한 4개의 데이터이다.
    1번과 4번, 두개의 데이터를 보면 다른 종류의 꽃임에도 값의 차이가 매우 적음.

### 개선 방향
- validation set을 이용한 training을 추가할 것.
- 강좌로 배운 내용을 적용해 볼 것
- 4개의 불순한 데이터에 대한 처리방법 고려

### Neural Net, xavier_initializer(), Drop out, AdamOptimizer 적용
    W shape를 [4,3]에서 [4,16] [16,256] [256,256] [256,256] [256,16] [16,3]로 바꿨을 때는 정확도가 오히려 이전보다 낮게나옴.
    
    Why?
[4,256] [256,256] [256,256] [256,256] [256,256] [256,3]으로 변경했을 때의 결과

![a](https://user-images.githubusercontent.com/31649100/52547477-5e1d3080-2e0b-11e9-9b0a-af05f7ef88de.png)

![b](https://user-images.githubusercontent.com/31649100/52547498-8b69de80-2e0b-11e9-87cf-6cc5bafe3da6.png)

    성능은 향상되었지만 테스트 결과는 이전과 같다.
    