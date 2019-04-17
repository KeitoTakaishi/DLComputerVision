# -*- coding: utf8-*-
from keras import Sequential, Model
from keras import Input
from keras import layers

'''
#テンソル
input_tensor = Input(shape=(32, ))
#関数
dense = layers.Dense(32, activation='relu')
#テンソルで呼び出された層はテンソルを返す
output_tensor = dense(input_tensor)
'''
#Sequential
'''
seq_model = Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))
'''

#-----------------------------------------------------------------------
#model
input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)

#入力、出力テンソルを元にモデルを作成する
model = Model(input_tensor, output_tensor)
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
#-----------------------------------------------------------------------
import numpy as np
x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))

model.fit(x_train, y_train, epochs=10, batch_size=128)
score = model.evaluate(x_train, y_train)
print(score)
