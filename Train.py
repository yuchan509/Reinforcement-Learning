#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import random
import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()


# #### 학습에 필요한 설정값을 정의

# In[2]:


# Epsilon 값.
epsilon = 1

# Epsilon 최소값.
epsilonMinimumValue = .001

# 에이전트의 행동 개수(좌로 움직이기, 가만히 있기, 우로 움직이기)
num_actions = 3

# 학습 반복 횟수.
num_epochs = 1000

# 은닉층의 개수.
hidden_size = 128

# Replay Memory 크기.(과거 행위를 기억하기 위한 공간)
maxMemory = 500

# batch size .
batch_size = 50

# 환경 크기.(게임판 크기)
gridSize = 10

# 게임 환경의 현재 상태.(10 x10)
state_size = gridSize * gridSize

# 규제강도.
discount = 0.9

# 학습률.
learning_rage = 0.2


# #### 시작과 끝값을 기준으로 랜덤값을 추출하는 함수를 정의

# In[3]:


def randf(s, e) :
    return (float(random.randrange(0, (e - s) * 9999)) / 10000) + s


# #### DQN 모델을 정의

# In[4]:


def build_DQN(x) :
    
    # 표준화를 위한 계수.
    a1 = 1.0 / math.sqrt(float(state_size))
    W1 = tf.Variable(tf.truncated_normal(shape=[state_size, hidden_size], stddev=a1))
    b1 = tf.Variable(tf.truncated_normal(shape=[hidden_size], stddev=0.01))
    H1_output = tf.nn.relu(tf.matmul(x, W1) + b1)
    
    a2 = 1.0 / math.sqrt(float(hidden_size))
    W2 = tf.Variable(tf.truncated_normal(shape=[hidden_size, hidden_size], stddev=a2))
    b2 = tf.Variable(tf.truncated_normal(shape=[hidden_size], stddev=0.01))
    H2_output = tf.nn.relu(tf.matmul(H1_output, W2) + b2)
    
    W3 = tf.Variable(tf.truncated_normal(shape=[hidden_size, num_actions], stddev=a2))
    b3 = tf.Variable(tf.truncated_normal(shape=[num_actions], stddev=0.01))
    output_layer = tf.matmul(H2_output, W3) + b3
    
    return tf.squeeze(output_layer)


# In[5]:


# 입력 화면 이미지와 타켓 Q값을 받기 위한 플레이스 홀더(메모리상의 저장공간)를 선언.
x = tf.placeholder(tf.float32, shape=[None, state_size])
y = tf.placeholder(tf.float32, shape=[None, num_actions])


# In[6]:


# DQN 모델을 선언하고 예측 결과를 리턴.
y_pred = build_DQN(x)


# In[7]:


# MSE 손실 함수와 옵티마이저를 정의.
loss      = tf.reduce_sum(tf.square(y-y_pred)) / (2 * batch_size)
optimizer = tf.train.GradientDescentOptimizer(learning_rage).minimize(loss)


# #### 게임 환경 구현

# In[8]:


class CatchEnvironment() :
    
    # 상태의 초기값을 지정.
    def __init__(self, gridSize) :
        
        # 그리드 개수.
        self.gridSize = gridSize
        
        # 입력 데이터 개수(그리드 개수 * 그리그 개수)
        self.state_size = self.gridSize * self.gridSize
        
        # 결과를 담을 행렬.
        self.state = np.empty(3, dtype = np.uint8)
        
        
    # 관찰 결과를 리턴.
    def observe(self) :
        # 현재 게임 화면 상태를 받아오기.
        canvas = self.drawState()
        
        # 1차원 행렬로 변환.
        canvas = np.reshape(canvas, (-1, self.state_size))
        
        return canvas
    
    # 게임 현재 상태를 계산.
    def drawState(self) :
        
        # 입력 데이터의 수 만큼의 행렬 계산.
        canvas = np.zeros((self.gridSize, self.gridSize))
        
        # 과일을 설정.
        canvas[self.state[0] - 1, self.state[1] - 1] = 1
        
        # 바구니를 설정.
        canvas[self.gridSize - 1, self.state[2] - 1 - 1] = 1
        canvas[self.gridSize - 1, self.state[2] - 1] = 1
        canvas[self.gridSize - 1, self.state[2] - 1 + 1] = 1
        
        return canvas
    
    # 게임을 초기상태로 리셋.
    def reset(self) :
        
        # 초기 과일 위치 초기화.
        initialFruitColumn    = random.randrange(1, self.gridSize + 1)
        
        # 초기 바구니 위치 초기화.
        initialBucketPosition = random.randrange(2, self.gridSize + 1 - 1)
        
        # 현재 상태 담기.
        self.state            = np.array([1, initialFruitColumn, initialBucketPosition])
        
        return self.getState()
    
    # 현재 상태를 불러오기.
    def getState(self) :
        stateInfo = self.state
        
        # 과일의 세로 위치.
        fruit_row = stateInfo[0]
        # 과일의 가로 위치.
        fruit_col = stateInfo[1]
        # 바구니의 가로 위치.
        basket = stateInfo[2]
        
        return fruit_row, fruit_col, basket
    
    
    # 에이전트가 취한 행동에 대한 보상을 줌.
    def getReward(self) :
        
        # 각 위치값의 위치를 가져오기.
        fruitRow, fruitCol, basket = self.getState()
        
        # 과일이 바닥에 닿았을 경우...
        if (fruitRow == self.gridSize - 1) :
            
            # 바구니가 과일을 받았다면 보상을 1로 반환.
            if (abs(fruitCol - basket) <= 1) :
                return 1
            
            # 과일을 받지 못했다면 보상을 -1로 반환.
            else :
                return -1
            
        # 과일이 바닥에 닿지 않았을 경우...
        else :
            # 아직 바닥에 닿지 않았으므로 보상을 0으로 반환.
            return 0
        
    # 게임이 끝났는지 확인.(1판 종료)
    def isGameOver(self) :
        # 과일이 바닥에 닿았는지 검사.
        if self.state[0] == self.gridSize - 1 :
            return True
        else :
            return False
        
        
    # action(좌, 제자리, 우)에 따라 바구니와 과일의 위치를 수정.
    def updateState(self, action) :
        move = 0
        if action == 0 :
            move = -1
        elif action == 1 :
            move = 0
        elif action == 2 :
            move = 1
            
        # 현재 과일과 바구니 위치를 가져오기.
        fruitRow, fruitCol, basket = self.getState()
        
        # 바구니의 위치를 업데이트.(min, max는 grid 밖으로 벗어나는 것을 방지)
        newBasket = min(max(2, basket + move), self.gridSize - 1)
        
        # 과일은 아래로 한칸 내리기.
        fruitRow = fruitRow + 1
        
        # 현재 상태로 다시 설정.
        self.state = np.array([fruitRow, fruitCol, newBasket])
        
    # 행동 수행.
    def act(self, action) :
        # Action 에 따라 현재 상태를 업데이트.
        self.updateState(action)
        
        # 업데이트된 상태를 보고 보상을 결정.
        reward = self.getReward()
        
        # 현재 게임 한판이 끝났는지 확인.
        gameOver = self.isGameOver()
        
        return self.observe(), reward, gameOver, self.getState()


# In[9]:


class ReplayMemory() :
    def __init__(self, gridSize, maxMemory, discount) :
        
        # 초기값 설정.
        
        # 사용할 최대 메모리량.
        self.maxMemory  = maxMemory
        # 게임 환경의 가로 세로 칸의 수.
        self.gridSize   = gridSize
        # 가로 * 세로 칸의 수.
        self.state_size = self.gridSize * self.gridSize
        # 규제 강도.
        self.discount   = discount
        
        # 게임 데이터를 담을 행렬 생성.
        canvas = np.zeros((self.gridSize * self.gridSize))
        canvas = np.reshape(canvas, (-1, self.state_size))
        
        # 입력 데이터를 담을 행렬 생성.
        self.inputState = np.empty((self.maxMemory, 100), dtype = np.float32)
        # 에이전트의 행동 데이터를 담을 행렬.
        self.actions    = np.zeros(self.maxMemory, dtype = np.uint8)
        # 에이전트가 행동을 취한 다음의 게임 상태를 담을 행렬.
        self.nextState  = np.empty((self.maxMemory, 100), dtype = np.float32)
        # 게임 오버 여부.
        self.gameOver   = np.empty(self.maxMemory, dtype = np.bool)
        # 보상.
        self.rewards    = np.empty(self.maxMemory, dtype = np.int8)
        
        # 플레이 횟수.
        self.count   = 0
        # 현재 보상의 결과.
        self.current = 0
        
    # 경험 저장.
    def remember(self, currentState, action, reward, nextState, gameOver) :
        self.actions[self.current]         = action
        self.rewards[self.current]         = reward
        self.inputState[self.current, ...] = currentState
        self.nextState[self.current, ...]  = nextState
        self.gameOver[self.current]        = gameOver
        self.count                         = max(self.count, self.current + 1)
        self.current                       = (self.current + 1) % self.maxMemory
        
    # 입력과 학습을 준비.
    def getBatch(self, y_pred, batch_size, num_actions, state_size, sess, X) :
        # 위에서 설정한 Batch size와 최대 메모리를 비교하여 더 작은 값을 구함.
        memoryLength    = self.count
        chosenBatchSize = min(batch_size, memoryLength)

        # 입력과 결과 데이터를 담을 행렬 생성.
        inputs  = np.zeros((chosenBatchSize, state_size))
        targets = np.zeros((chosenBatchSize, num_actions))
        
         # 배치 안에서 값을 추출하여 담기.
        for i in range(chosenBatchSize) :
            # 배치에 포함될 기억을 랜덤하게 선택.
            randomIndex = random.randrange(0, memoryLength)
            
             # 현재 상태와 Q값을 가져옴.
            current_inputState = np.reshape(self.inputState[randomIndex], (1, 100))
            target             = sess.run(y_pred, feed_dict = {X : current_inputState})

            # 현재 상태 바로 다음 상태를 불러오고 다음 상태에서 취할 수 있는 가장 큰 Q값을 계산.
            current_nextState  = np.reshape(self.nextState[randomIndex], (1, 100))
            nextStateQ         = sess.run(y_pred, feed_dict = {X : current_nextState})
            nextStateMaxQ      = np.amax(nextStateQ)
            
            # 게임 오버일시 보상으로 Q값을 업데이트.
            if (self.gameOver[randomIndex] == True):
                target[self.actions[randomIndex]] = self.rewards[randomIndex]
            else:
                target[self.actions[randomIndex]] = self.rewards[randomIndex] + self.discount * nextStateMaxQ

            # 입력과 결과 데이터에 값을 저장.
            inputs[i]  = current_inputState
            targets[i] = target

        # 결과 리턴.
        return inputs, targets


# #### Tensorflow를 통해 학습 가동시 자동으로 호출되는 함수

# In[10]:


def main(_) :
    print('학습을 시작하겠습니다.')
    
    # 게임 플레이 환경 선언.
    env = CatchEnvironment(gridSize)
    
    # ReplayMemory 선언.
    memory = ReplayMemory(gridSize, maxMemory, discount)
    
    # 학습된 파라미터를 저장하기 위한 Saver 선언.
    saver = tf.train.Saver()
    
    # 과일을 받은 수.
    winCount = 0
    
    # With 문을 통해 작업이 완료되면 Tensorflow가 자동 종료.
    with tf.Session() as sess :
        # 변수들 초기값 할당.
        sess.run(tf.global_variables_initializer())
        
        # 학습 횟수 만큼 반복.
        for i in range(num_epochs + 1) :
            # 환경 초기화.
            err = 0
            env.reset()
            isGameOver = False
            
            # 최초 상태 가져오기.
            currentState = env.observe()
            
            # 과일이 바닥에 닿을 때까지 반복.
            while isGameOver != True :
                # Q값을 초기화.
                action = -9999
                global epsilon
                
                # 만약 0 ~ 1 사이의 임의값이 앱실론 값보다 작거나 같을 경우...
                if randf(0, 1) <= epsilon :
                    # 랜덤 행동 지정.
                    action = random.randrange(0, num_actions)
                # 클 경우...
                else :
                    # 각 행동에 대한 Q값을 계산.
                    q = sess.run(y_pred, feed_dict = {x : currentState})
                    
                    # q가 가장 큰 행동을 담기.
                    action = q.argmax()
                
                # 앱실론 값에 0.999 값을 곱해서 앱실론 값을 조정.
                if epsilon > epsilonMinimumValue :
                    epsilon = epsilon * 0.999
                
                # 에이전트가 구한 행동을 통해 보상과 다음 상태를 가져옴.
                nextState, reward, gameOver, stateInfo = env.act(action)
                
                # 만약 과일을 받아냈다고 한다면, 점수를 1 증가.
                if reward == 1:
                    winCount = winCount + 1
                    
                # 에이전트가 행동한 결과를 replayMemory에 저장.
                memory.remember(currentState, action, reward, nextState, gameOver)
                
                # 다음 이동을 위해 현재 상태를 재설정.
                currentState = nextState
                
                # 게임 오버 여부를 담기.
                isGameOver = gameOver
                
                # ReplayMemory로 부터 학습에 사용할 Batch 데이터를 가져오기.
                inputs, targets = memory.getBatch(y_pred, batch_size, num_actions, state_size, sess, x)
                
                # 최적화(가중치 수정)을 수행하고, 손실함수를 반환.
                _, loss_print = sess.run([optimizer, loss], feed_dict = {x : inputs, y : targets})
                
                # 손실율 누적.
                err += loss_print
            
            a100 = (float(winCount) / float(i + 1)) * 100
            print(f'반복 : {i}, 에러 : {err}, 승리 : {winCount}, 승리비율 : {a100}')
        
        print('학습이 완료되었습니다.')
        save_path = saver.save(sess, 'model.ckpt')
        print('모델을 저장하였습니다.')


# In[11]:


if __name__ == '__main__' :
    # main 함수 호출.
    tf.app.run()

