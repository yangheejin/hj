import pandas as pd #csv파일 읽기
import numpy as np # 행렬연산
import matplotlib.pyplot as plt #데이터 시각화
from keras.models import Sequential #keras는 딥러닝에 필요한 라이브러리 #Sequential은 딥러닝 모델
from keras.layers import LSTM, Dropout, Dense, Activation
import datetime

data = pd.read_csv('dataset/005930.KS_5y.csv') #pandas의 read라는 함수로 csv파일을 불러옴
high_prices = data['High'].values #불러온 데이터에서 높은 가격과
low_prices = data['Low'].values # 낮은 가격을 합쳐 2로 나누면
mid_prices = (high_prices + low_prices) /2 # 중간가격이 나옴


# window를 만든다 result라는 배열에 50개의 mid prices가 들어감
seq_len = 50
sequence_length = seq_len +1 #최근 50일간의 데이터를 보고 내일의 데이터(51번째 데이터)를 예측
result=[] #result란 리스트를 생성
for index in range(len(mid_prices)-sequence_length):
  result.append(mid_prices[index:index + sequence_length])
 #result란 리스트에 50개의 평균값(mid_prices)를 넣어줌


#정규화
#정규화를 해야 정규분포에서 쓸 수 있는 여러 수학적 계산들을 컴퓨터가 수행할 수 있어서 정규화를 함.
normalized_data=[]
for window in result:
  normalized_window = [((float(p) / float(window[0]))-1) for p in window]
  normalized_data.append(normalized_window)
result = np.array(normalized_data) #정규화를 끝낸 배열을 다시 result에 넣음(덮어쓰기)



# 학습을 시킬 training set 와 제대로 작동하는지 확인할 test set으로 나눠야함
row = int(round(result.shape[0]*0.9)) # 90퍼센트는 traininng set 10퍼센트는 test set임
train = result[:row,:] # training set
np.random.shuffle(train) # training set를 순서대로 학습하는 게 아니고 랜덤으로 train을 섞은 다음 학습할거라서 random함수로 섞음

x_train = train[:,:-1] #50개를 넣음 x축은 날짜
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:,-1] #1개를 넣음 y축은 주식가격

x_test = result[row:, :-1] # 예측
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
y_test = result[row:, -1] # 예측

#print(x_train.shape,x_test.shape) #시험 출력

model = Sequential() #LSTM이라는 머신러닝 모듈이 아예 존재해서 이용하기만 하면 됨
#LSTM에 내장된 Sequential 클래스로 모델을 생성
model.add(LSTM(50,return_sequences=True, input_shape=(50,1))) # 입력은 50개가 들어감
model.add(LSTM(64,return_sequences=False)) # 데이터 입력이 잘못된 경우. 64란 숫자는 아무거나 넣어도 상관없음
model.add(Dense(1,activation='linear')) #1개의 출력값이 나옴
model.compile(loss='mse',optimizer = 'rmsprop') #계산 중 생기는 손실 관련 코드


model.fit(x_train,y_train,
  validation_data=(x_test, y_test), # test에 넣을 x와 y를 입력(x:날짜, y:주식 가격)
  batch_size=10, #한번에 몇개씩 묶어서 학습시킬것인지 정함 10개씩 묶어서 학습시킨다는 뜻
  epochs=3 #3번 동안 학습을 반복해서 정확성을 높임
)

pred = model.predict(x_test) #모델을 이용해 내일의 값을 예측


#마지막으로 꺾은선 그래프로 예측한 데이터를 표시한다
fig = plt.figure(facecolor='white') # 배경을 흰색으로 설정
ax = fig.add_subplot(111) #일반적인 레이아웃으로 설정
ax.plot(y_test,label='True')# plot 메소드는 주어진 데이터를 꺾은선그래프로 보여준다.
ax.plot(pred,label='Prediction') 
plt.show() #그래프 표시