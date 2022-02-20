import numpy
from imblearn.over_sampling import SMOTE
from numpy import savetxt
from numpy import loadtxt
from matplotlib import pyplot
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D,AveragePooling1D
from numpy import mean
from tensorflow.random import set_seed
import tensorflow
from keras.constraints import min_max_norm
from keras.regularizers import L2
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import RobustScaler
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# setting the seed
seed(1)
set_seed(1)

# load training data
#X_train_whole = loadtxt('d:\\a_eeg_1_to_45\\EEGData_512.csv', delimiter=',')

# load the test data
X = loadtxt('d:\\a_eeg_1_to_45\\EEGData_512_Test_New.csv', delimiter=',')

# combine the train and test data
#data_combined = numpy.append(X_train_whole, X, axis=0)
data_combined = X

# shuffle the combined data
numpy.random.seed(2) 
numpy.random.shuffle(data_combined)
print(data_combined.shape)

index1 = 398 + 0  #199
index2 = 0        #199

# Divide 398 + 199 shuffled samples for training and 199 shuffled samples for testing. This will mix the readings taken in 2 different batches. 
savetxt('d:\\a_eeg_1_to_45\\shuffled_test.csv', data_combined[index1:index1+index2, :], delimiter=',') 

# split the training data between training and validation
tensorflow.compat.v1.reset_default_graph()
X_train_tmp, X_test_tmp, Y_train_tmp, Y_test_tmp = train_test_split(data_combined[0:index1, :], data_combined[0:index1, -1], random_state=1, test_size=0.3, shuffle = False)
print(X_train_tmp.shape)
print(X_test_tmp.shape)


# augment train data
choice = X_train_tmp[:, -1] == 0.
X_total_1 = numpy.append(X_train_tmp, X_train_tmp[choice, :], axis=0)
X_total_2 = numpy.append(X_total_1, X_train_tmp[choice, :], axis=0)
X_total_3 = numpy.append(X_total_2, X_train_tmp[choice, :], axis=0)
X_total_4 = numpy.append(X_total_3, X_train_tmp[choice, :], axis=0)
X_total = numpy.append(X_total_4, X_train_tmp[choice, :], axis=0)


print(X_total.shape)

# data balancing for train data
sm = SMOTE(random_state = 2)
X_train_keep, Y_train_keep = sm.fit_resample(X_total, X_total[:, -1].ravel())
print("After OverSampling, counts of label '1': {}".format(sum(Y_train_keep == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(Y_train_keep == 0)))


#=======================================
 
# Data Pre-processing - presently nothing

input = X_train_keep[:, 0:576]
testinput = X_test_tmp[:,0:576]
Y_train = X_train_keep[:, -1]
Y_test = X_test_tmp[:, -1]

#=====================================

# Model configuration

print(len(input))
print(len(testinput))

input = input.reshape(len(input), 9, 64)
#input = input.transpose(0, 2, 1)
print (input.shape)

testinput = testinput.reshape(len(testinput), 9, 64)
#testinput = testinput.transpose(0, 2, 1)
print (testinput.shape)

# Create the model
model=Sequential()
model.add(Conv1D(filters=80, kernel_size=4, kernel_regularizer=L2(0.003), bias_regularizer=L2(0.003), activity_regularizer = L2(0.003), kernel_constraint=min_max_norm(min_value=-0.1, max_value=0.1), data_format='channels_last', padding='valid', activation='relu', strides=1, input_shape=(9, 64)))
model.add(Dropout(0.4))
model.add(Conv1D(filters=40, kernel_size=4, kernel_regularizer=L2(0.003), bias_regularizer=L2(0.003), activity_regularizer = L2(0.003), kernel_constraint=min_max_norm(min_value=-0.1, max_value=0.1), data_format='channels_last', padding='valid', activation='relu', strides=1))
model.add(Dropout(0.4))
model.add(AveragePooling1D(pool_size=2))
model.add(Dense(2, activation='softmax'))

model.summary()

# Compile the model
sgd = optimizers.SGD(lr=0.006, momentum=0.7, nesterov=True)       
model.compile(loss=sparse_categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

# early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=400)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

hist = model.fit(input, Y_train, batch_size=24, epochs=500, verbose=1, validation_data=(testinput, Y_test), steps_per_epoch=None, callbacks=[es, mc])

# evaluate the model
Y_hat_classes = model.predict_classes(testinput)
matrix = confusion_matrix(Y_test, Y_hat_classes)
print(matrix)


# plot training and validation history
pyplot.plot(hist.history['loss'], label='tr_loss')
pyplot.plot(hist.history['val_loss'], label='val_loss')
pyplot.plot(hist.history['accuracy'], label='tr_accuracy')
pyplot.plot(hist.history['val_accuracy'], label='val_accuracy')
pyplot.legend()
pyplot.xlabel("No of iterations")
pyplot.ylabel("Accuracy and loss")
pyplot.show()

#==================================

model.save("D:\\a_eeg_1_to_45\\model_conv1d.h5")

# load the saved model
saved_model = load_model('best_model.h5')
# evaluate the model
_, train_acc = saved_model.evaluate(input, Y_train, verbose=1)
_, test_acc = saved_model.evaluate(testinput, Y_test, verbose=1)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

#==================================

#Removed dropout and reduced momentum and reduced learning rate