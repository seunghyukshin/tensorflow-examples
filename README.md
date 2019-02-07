# tensorflow-examples

## 1. 피파19 선수들의 능력치를 이용한 Linear regression 예제

### 가설
Age(나이),Overall(통합 능력치),Potential(잠재 능력치)를 입력하여 Value(몸값)을 예측한다.

### 결과
 예측 할 수 없는 값

### 실패 요인
가설이 잘 못 되었다.\
나이,오버롤,포텐셜 외에 변수를 추가해준다 하더라도 예측했던 값이 나오지 않을 가능성 높음\
**Linear하다는 근거가 없이 MLR식에 넣었기 때문**

### 개선 방향
- normalized하여 cost값을 줄인다.\
inverse_transform 
  
- 변수를 줄여 linear형태를 찾는다.\
→ x:Overall, y:Value\
→애초에 데이터셋이 좋지 않았다.
![figure_1](https://user-images.githubusercontent.com/31649100/52390948-51809b80-2ade-11e9-82f0-2c91a2f8c9f3.png)\
ovearll에 해당하는 value값들 중 **최댓값 혹은 평균값**을 택한다면 데이터를 다루기 쉬울 것으로 보임.
