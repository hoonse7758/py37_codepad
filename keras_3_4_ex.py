# -*- coding: utf-8 -*-

# import
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import imdb
from keras import models
from keras import layers
from keras import activations
from keras import optimizers
from keras import losses
from keras import metrics

from websocket import server

import time
import tensorflow as tf

import tornado.web
from tornado import gen
from tornado.ioloop import PeriodicCallback, IOLoop
from tornado.websocket import websocket_connect

from websocket.server import WebSocketHandler


class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    DIRECTORY = './logs'
    port = 1234

    def __init__(self):
        super(LossAndErrorPrintingCallback, self).__init__()

        # # WebSocket
        # self._app = tornado.web.Application([
        #     (r"/", WebSocketHandler),
        # ])
        # self._app.listen(self.port)
        # tornado.ioloop.IOLoop.instance().start()


        # WebSocket URL
        self.url = f"ws://localhost:{self.port}"

        # Message Sender
        self.ioloop = IOLoop.instance()
        self.ws = None
        self.connect()
        PeriodicCallback(self.keep_alive, 20000).start()
        self.ioloop.start()

    @gen.coroutine
    def connect(self):
        print("trying to connect socket")
        try:
            self.ws = yield websocket_connect(self.url)
        except Exception as e:
            print("connection error")
        else:
            print("connected message sender to socket")

    def keep_alive(self):
        if self.ws is None:
            self.connect()
        else:
            self.ws.write_message("Sender Listening")

    # @gen.coroutine
    # def run(self):
    #     once = False
    #     while True:
    #         msg = yield self.ws.read_message()
    #         # print(msg)
    #         if once:
    #             time.sleep(11)
    #             once = False
    #         else:
    #             time.sleep(1)
    #             once = True
    #         self.ws.write_message("Hello matey")
    #         if msg is None:
    #             print("connection closed")
    #             self.ws = None
    #             break

    # @gen.coroutine
    def on_train_begin(self, logs=None):
        print("train start")
        # # WebSocket
        # self._app = tornado.web.Application([
        #     (r"/", WebSocketHandler),
        # ])
        # self._app.listen(self.port)
        # tornado.ioloop.IOLoop.instance().start()
        #
        # # WebSocket URL
        # self.url = f"ws://localhost:{self.port}"
        #
        # # Message Sender
        # self.ws = None
        # self.connect()
        # PeriodicCallback(self.keep_alive, 20000).start()
        # # self.ioloop.start()

    # @gen.coroutine
    def on_test_begin(self, logs=None):
        print("test start")

    # @gen.coroutine
    def on_predict_begin(self, logs=None):
        print("predict start")

    # @gen.coroutine
    def on_train_batch_end(self, batch, logs=None):
        msg = yield self.ws.read_message()
        train_log = open(self.DIRECTORY + r'/' + 'output' + r'/' + 'train_log.txt', 'a')
        log = 'For batch {}, loss is {:7.2f}.'.format(batch, logs['loss'])
        train_log.write(log)
        train_log.write('\n')
        train_log.close()
        self.ws.write_message(log)
        print(log)

    # @gen.coroutine
    def on_test_batch_end(self, batch, logs=None):
        msg = yield self.ws.read_message()
        train_log = open(self.DIRECTORY + r'/' + 'output' + r'/' + 'train_log.txt', 'a')
        log = 'For batch {}, loss is {:7.2f}.'.format(batch, logs['loss'])
        train_log.write(log)
        train_log.write('\n')
        train_log.close()
        self.ws.write_message(log)
        print(log)

    # @gen.coroutine
    def on_epoch_end(self, epoch, logs=None):
        msg = yield self.ws.read_message()
        train_log = open(self.DIRECTORY + r'/' + 'output' + r'/' + 'train_log.txt', 'a')
        log = 'The average loss for epoch {} is {:7.2f}.'.format(epoch, logs['loss'])
        train_log.write(log)
        train_log.write('\n')
        train_log.close()
        self.ws.write_message(log)
        print(log)
        print('============================================================')


# Loading the IMDB dataset
train, test = imdb.load_data(num_words=10000)   # 가장 많이 언급된 Top 10,000 단어만 변환된 데이터 ( 그 이하로 언급된 단어들은 빠짐 )
train_data, train_labels = train
test_data, test_labels = test


# Decode 해보기 - data 를 원래 글로 복원하기
word_encoder = imdb.get_word_index()  # 단어 인코딩에 사용된 규칙 dictionary {단어: 번호}
word_decoder = dict(zip(word_encoder.values(), word_encoder.keys()))    # 디코딩 규칙 {번호: 단어}


# train_data[0] 에 대한 복원 결과 - 데이터 이해를 돕기 위함
decoded_sample = ' '.join(list(map(lambda index: word_decoder.get(index - 3, '?'), train_data[0])))


# Encoding the integer sequences into a binary matrix
def vectorize_sequences(sequences, dimension=10000):
    """
    ( 데이터 개수 x 출현한 단어 가짓수 ) 사이즈의 행렬로 반환
    반환되는 행렬의 ( i, j ) 원소의 의미는
    results[i, j] = 1 => i 번째 리뷰에는 인덱스가 j 인 단어가 언급되어 있음
    results[i, j] = 0 => i 번째 리뷰에는 인덱스가 j 인 단어가 언급되어 있지 않음
    :param sequences: 원래 input numpy array
    :param dimension: 데이터 내 출현하는 단어들의 가짓수
    :return:
    """
    results = np.zeros((len(sequences), dimension))     # 데이터 갯수 X 차원 수 사이즈 0 행렬 생성
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# 실제 아래와 같이 변환 수행하면 다음의 관계가 성립함
# sorted(np.where(train_x[i] == 1.)[0].tolist()) == sorted(set(train_data))
train_x = vectorize_sequences(train_data)
test_x = vectorize_sequences(test_data)

# no_for_loop = list(
#     map(lambda idxs: list(map(lambda i: float(i in idxs),
#                               range(10000))),
#         map(lambda seq: sorted(set(seq)),
#             train_data)))

# 라벨은 이미 0 아니면 1만 가지므로 복잡한 처리가 필요 없음
# 실행 결과로 각 array 별로 원소 타입만 float32 로 변환됨
train_y = np.asarray(train_labels).astype('float32')
test_y = np.asarray(test_labels).astype('float32')


# train 세트를 다시 train / validation 용으로 분할
val_x = train_x[:10000]     # validation 용 세트 - 앞에 10,000 개를 사용
val_y = train_y[:10000]
train_x = train_x[10000:]   # train 용 세트 - 뒤에 나머지 15,000 개를 사용
train_y = train_y[10000:]


# The Model Definition
model = models.Sequential()
model.add(layers.Dense(units=16, activation=activations.relu, input_shape=train_x.shape[1:]))
model.add(layers.Dense(units=16, activation='relu'))
model.add(layers.Dense(units=1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,      # 'binary_crossentropy',  문자열로 줘도 작동
              metrics=[metrics.binary_accuracy])    # 'accuracy' 문자열로 줘도 작동

# Training the Model
train_history = model.fit(x=train_x, y=train_y,
                          epochs=20, batch_size=512,
                          validation_data=(val_x, val_y),
                          callbacks=[LossAndErrorPrintingCallback()])


# Plotting the Result - epoch 별 손실 값 곡선
model_measures = train_history.history
losses = model_measures.get('loss')
val_losses = model_measures.get('val_loss')
epochs = range(1, len(losses) + 1)
plt.plot(epochs, losses, 'bo', label='Training Loss')
plt.plot(epochs, val_losses, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Plotting the Result - epoch 별 정확도 값 곡선
accs = model_measures.get('acc')
val_accs = model_measures.get('val_acc')
epochs = range(1, len(losses) + 1)
plt.clf()   # 기존 plot 지우기
plt.plot(epochs, losses, 'bo', label='Training Accuracy')
plt.plot(epochs, val_losses, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Using Trained Network to Predict New Data
test_pred = model.predict(x=test_x)


print("Debug Point")
