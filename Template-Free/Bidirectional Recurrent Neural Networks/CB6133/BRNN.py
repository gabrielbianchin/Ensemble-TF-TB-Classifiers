from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, Dense, Input, concatenate, Flatten, Reshape, Dropout, Embedding
from tensorflow.compat.v1.keras.layers import CuDNNGRU
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def createNN(qt, emb=1):
  x = Input(shape=(700, 21, ))
  flat = Flatten()(x)
  y = Embedding(21, emb, input_length=(14700,))(flat)
  flat = Flatten()(y)
  re = Reshape((700, 21 * emb))(flat)

  x_dist = Input(shape=(700, 21))

  conc = concatenate([re, x_dist])
  
  gru = Bidirectional(CuDNNGRU(600, return_sequences=True))(conc)
  
  for _ in range(qt):
    gru = Bidirectional(CuDNNGRU(600, return_sequences=True))(gru)

  output = Dense(8, activation='softmax')(gru)


  model = Model([x, x_dist],output)

  optimizer = tf.keras.optimizers.Adam(lr=0.0001)

  model.compile(optimizer = optimizer, metrics = ['acc'], loss='categorical_crossentropy')

  return model


base = np.load('/content/drive/My Drive/cullpdb+profile_6133.npy')
base = np.reshape(base, (-1, 700, 57))
a = np.arange(0,21)
b = np.arange(35,56)
c = np.hstack((a,b))

features = base[:, :, a]
classes = base[:, :, 22:30]

featuresinv = np.fliplr(features)
classesinv = np.fliplr(classes)

X_train = features[:5600,:,:]
X_val = features[5877:6133,:,:]
X_test = features[5605:5877, :, :]

y_train = classes[:5600,:,:]
y_val = classes[5877:6133,:,:]
y_test = classes[5605:5877,:,:]

X_train_inv = featuresinv[:5600,:,:]
X_val_inv = featuresinv[5877:,:,:]
X_test_inv = featuresinv[5605:5877,:,:]

y_train_inv = classesinv[:5600,:,:]
y_val_inv = classesinv[5877:,:,:]
y_test_inv = classesinv[5605:5877,:,:]



def train_and_evaluate_model(model, x_t, x_t_d, y_t, x_v, x_v_d, y_v, x_tst, x_tst_d, y_tst, option, qtde):
    
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='min', verbose=1)
    model.fit([x_t, x_t_d], y_t, epochs=50, batch_size=32, verbose = 1, validation_data = ([x_v, x_v_d], y_v), callbacks = [es, lr])
    
    predicted = model.predict([x_tst, x_tst_d])

    y_teste = []
    predict = []
    
    predfinal = []
    y_testfinal = []
    
    predicted = np.reshape(predicted, (predicted.shape[0] * predicted.shape[1], 8))
    y_tst = np.reshape(y_tst, (y_tst.shape[0] * y_tst.shape[1], 8))
    x_tst = np.reshape(x_tst, (x_tst.shape[0] * x_tst.shape[1], x_tst.shape[2]))
    print(predicted[:1])
    for i in range(len(x_tst)):
        cont = 0
        for j in range(len(x_tst[i])):
            cont += x_tst[i][j]
        if cont != 0:
            y_teste.append(y_tst[i])
            predict.append(predicted[i])
        if (i+1) % 700 == 0:
            if option == 2:
                y_teste = y_teste[::-1]
                predict = predict[::-1]
            for k in range(len(predict)):
                y_testfinal.append(y_teste[k])
                predfinal.append(predict[k])
            y_teste = []
            predict = []

    return np.asarray(predfinal), np.asarray(y_testfinal)

for i in range(5):
    print(i+1)
    model = None
    model = createNN(i+1)
    model.summary()
    p1, e1 = train_and_evaluate_model(model, X_train, X_train_dist, y_train, X_val, X_val_dist, y_val, X_test, X_test_dist, y_test, 1, i+1)
    np.save('/content/drive/My Drive/embedding/Results/pred-cb6133-BRNN-' + str(i+1) + '-option1.npy', p1)
    np.save('/content/drive/My Drive/embedding/Results/pred-cb6133-BRNN-' + str(i+1) + '-option1.npy', e1)

    model = None
    model = createNN(i+1)
    p2, e2 = train_and_evaluate_model(model, X_train_inv, X_train_dist_inv, y_train_inv, X_val_inv, X_val_dist_inv, y_val_inv, X_test_inv, X_test_dist_inv, y_test_inv, 2, i+1)
    np.save('/content/drive/My Drive/embedding/Results/pred-cb6133-BRNN-' + str(i+1) + '-option2.npy', p2)
    np.save('/content/drive/My Drive/embedding/Results/pred-cb6133-BRNN-' + str(i+1) + '-option2.npy', e2)