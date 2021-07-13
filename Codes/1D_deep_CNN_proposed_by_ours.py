import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential
from tensorflow.keras.layers import Conv1D, ZeroPadding1D, MaxPooling1D, Flatten, Dropout, Dense, BatchNormalization, Activation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import datetime

import sklearn
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score, precision_recall_curve
import sklearn.metrics as sklearn_metrics
from itertools import cycle
from scipy.interpolate import interp1d
import seaborn as sns


np.set_printoptions(threshold=np.inf)
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_metrics(history):
    # Draw loss, AUC, precision and recall curve
    plt.figure(figsize=(8, 4), dpi=800)
    metrics =  ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch,  history.history[metric], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend()
        plt.tight_layout()

def plot_cm(labels, predictions):
    # Draw confusing matrix
    cm = confusion_matrix(labels, predictions, normalize='all')
    plt.figure()
    sns.heatmap(cm, cmap="Oranges")
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

def plot_prc(name, labels, predictions, **kwargs):
    # Draw Precision_Recall curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes=1332
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(labels[:, i], predictions[:, i])
        average_precision[i] = average_precision_score(labels[:, i], predictions[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(labels.ravel(), predictions.ravel())
    average_precision["micro"] = average_precision_score(labels, predictions, average="micro")
    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.4f}'.format(average_precision["micro"]))


def load_spectrum_file(dir_path):
    data = []
    label = []
    file_dir_list = os.listdir(dir_path)
    for file_dir in file_dir_list:
        file_list = os.listdir(dir_path + '/' + file_dir)
        for filename in file_list:
            file_path = dir_path + '/' + file_dir + '/' + filename
            x, y = np.loadtxt(file_path, dtype=float, comments='#', delimiter=',', unpack=True)
            data.append(y)
            label.append(file_dir)
    return data, label

model = Sequential()
model.add(keras.Input(shape=(1570, 1)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(2, strides=2))

model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(2, strides=2))

model.add(Conv1D(256, 3, activation='relu'))
model.add(Conv1D(256, 3, activation='relu'))
model.add(Conv1D(256, 3, activation='relu'))
model.add(Conv1D(256, 3, activation='relu'))
model.add(MaxPooling1D(2, strides=2))

model.add(Conv1D(512, 3, activation='relu'))
model.add(Conv1D(512, 3, activation='relu'))
model.add(Conv1D(512, 3, activation='relu'))
model.add(Conv1D(512, 3, activation='relu'))
model.add(MaxPooling1D(2, strides=2))

model.add(Conv1D(512, 3, activation='relu'))
model.add(Conv1D(512, 3, activation='relu'))
model.add(Conv1D(512, 3, activation='relu'))
model.add(Conv1D(512, 3, activation='relu'))
model.add(MaxPooling1D(2, strides=2))

model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(192, activation='softmax'))  # Dataset_1
# model.add(Dense(1332, activation='softmax')) #Dataset_2
model.summary()

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.CategoricalAccuracy(name='accuracy'),
    keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_accuracy'),
    keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
    keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR')
]
model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics= METRICS,
)

print("Loading data...")
data, label = load_spectrum_file('./Dataset_1')  # Dataset_1
# data, label = load_spectrum_file('./Dataset_2')  # Dataset_2
data = np.array(data)
data = tf.reshape(data, [data.shape[0], data.shape[1], 1])
lb = LabelBinarizer()
labels = lb.fit_transform(label)
train_data, test_data, train_label, test_label = train_test_split(np.array(data), labels, train_size = 0.7, test_size = 0.3, stratify=labels, shuffle=True)
print("Completed!")

# tf.keras.utils.plot_model(model, "./model_architecture_diagram.png", show_shapes=True)
# log_dir = "/ftp_server/SangXiancheng/logs/fit_vgg19/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', verbose=2, patience=30, mode='max', restore_best_weights=True)
h = model.fit(train_data, train_label, batch_size=32, epochs=200, verbose=1, validation_split=0.2, callbacks=[early_stopping])
plot_metrics(h)
results = model.evaluate(test_data,  test_label, batch_size=32, return_dict=True, verbose=2)
F1_score = 2 * (results['precision'] * results['recall']) / (results['precision'] + results['recall'])
print("loss: {:0.4f}".format(results['loss']))
print("accuracy: {:0.4f}".format(results['accuracy']))
print("F1_score: {:0.4f}".format(F1_score))
plt.savefig('./1D_deep_CNN_model_train_curve.jpeg')
plt.show()
test_predictions = model.predict(test_data, batch_size=32)

# BATCH_SIZE=32
# train_predictions_baseline = model.predict(train_data, batch_size=BATCH_SIZE)
# test_predictions_baseline = model.predict(test_data, batch_size=BATCH_SIZE)
# plot_prc("Train Baseline", train_label, train_predictions_baseline, color=colors[0])
# plot_prc("Test Baseline", test_label, test_predictions_baseline, color=colors[0], linestyle='--')
# plt.legend(loc='lower right')
# plt.show()
# plot_cm(lb.inverse_transform(test_label), lb.inverse_transform(test_predictions))
#
y_true = test_label
y_pred = test_predictions
for i in range(len(y_pred)):
    max_value=max(y_pred[i])
    if max_value == 0:
        print(i, ":")
        print("y_true:")
        print(y_true[i])
        print("y_pred:")
        print(y_pred[i])

    for j in range(len(y_pred[i])):
        if max_value==y_pred[i][j]:
            y_pred[i][j]=1
        else:
            y_pred[i][j]=0

print('Classification_report', sklearn_metrics.classification_report(y_true, y_pred, digits=4, zero_division=1))
print('accuracy_score', sklearn_metrics.accuracy_score(y_true, y_pred))
print('------Weighted------')
print('Weighted precision', precision_score(y_true, y_pred, average='weighted', zero_division=1))
print('Weighted recall', recall_score(y_true, y_pred, average='weighted', zero_division=1))
print('Weighted f1-score', f1_score(y_true, y_pred, average='weighted', zero_division=1))
