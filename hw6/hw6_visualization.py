import csv
import sys
import random
import keras
import numpy as np
import pandas as pd
from keras.layers import Dense, Input,Dropout
from keras.models import Model, load_model
from sklearn.cluster import KMeans

'''

image_path = 'C:/Users/user/Desktop/image.npy'
test_case_path = 'C:/Users/user/Desktop/test_case.csv'
prediction_path = 'C:/Users/user/Desktop/prediction.csv'
autoencoder_model_path = 'C:/Users/user/Dropbox/Course/gadurate courses/Machine Learning/hw6/hw6_autoencoder.h5'
encoder_model_path = 'C:/Users/user/Dropbox/Course/gadurate courses/Machine Learning/hw6/hw6_encoder.h5'
visualization_path = 'C:/Users/user/Desktop/visualization.npy'
'''
image_path = sys.argv[1]
test_case_path = sys.argv[2]
prediction_path = sys.argv[3]
autoencoder_model_path = 'hw6_autoencoder.h5'
encoder_model_path = 'hw6_encoder.h5'

EPOCHS = 100
BATCH_SIZE = 5000
VALID_SPLIT = 0.1
random.seed(24)

image = np.load(image_path)
test_case = pd.read_csv(test_case_path)

def evaluate_testing(cluster_labels,test_data = test_case ):
    I1 = [cluster_labels[i] for i in test_data['image1_index']]
    I2 = [cluster_labels[i] for i in test_data['image2_index']]
    result = []
    for i in range(len(I1)):
        result.append(1 * (I1[i] == I2[i]))
    return(result)

def write_result(prediction, prediction_path=prediction_path):
    text = open(prediction_path, "w+")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["ID", "ANS"])
    for i in range(len(prediction)):
        s.writerow([i, prediction[i]])
    text.close()

def build_model():
    input_image = Input(shape=(28 * 28,), name='InputImage')
    encoded = Dense(512, activation='relu')(input_image)
    encoded = Dense(256, activation='relu')(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoder_output = Dense(64)(encoded)
    decoded = Dense(256, activation='relu')(encoder_output)
    encoded = Dense(512, activation='relu')(encoded)
    output_image = Dense(28 * 28, activation='tanh')(decoded)
    # construct the autoencoder model
    autoencoder = Model(input=input_image, output=output_image)
    # construct the encoder model for plotting
    encoder = Model(input=input_image, output=encoder_output)
    encoder.summary()
    autoencoder.summary()
    return autoencoder, encoder
'''
autoencoder, encoder = build_model()

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(image / 255. - 0.5,
                image / 255. - 0.5,
                epochs = EPOCHS,
                batch_size = BATCH_SIZE,
                shuffle = True, validation_split = VALID_SPLIT)

	autoencoder.save(autoencoder_model_path)
	encoder.save(encoder_model_path)
'''	

autoencoder = load_model(autoencoder_model_path)
encoder = load_model(encoder_model_path)
encoder_images = encoder.predict(image / 255. - 0.5)
kmeans = KMeans(n_clusters=2, random_state=0).fit(encoder_images)
result = evaluate_testing(kmeans.labels_)
write_result(result)





visualization = np.load(visualization_path)

images_predict = encoder.predict(visualization / 255. - 0.5)


from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
a = []
for i in range(5000):
    a.append(0)

for i in range(5000):
    a.append(1)
y = np.array(a)

#perform t-SNE embedding
vis_data = TSNE(n_components=2).fit_transform(x)
kmeans = KMeans(n_clusters=2, random_state=0).fit(vis_data)
#plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]
cm = plt.cm.get_cmap('Dark2')
sc = plt.scatter(vis_x, vis_y, c=y, cmap=cm)
plt.colorbar(sc)
plt.show()
np.save('C:/Users/user/Desktop/vis_data.npy', vis_data)


import pylab as pl
for i in range(0, kmeans.labels_.shape[0]):
    if kmeans.labels_[i] == 0:
        c1 = plt.scatter(vis_x[i], vis_y[i], c='r', marker='+')
    elif kmeans.labels_[i] == 1:
        c2 = plt.scatter(vis_x[i], vis_y[i], c='g', marker='o')
plt.legend([c1, c2], ['0', '1'])
plt.show()




II = np.reshape(visualization[5000], (28, 28))
imgplot = plt.imshow(II)
plt.show()


'''
correct_answer_path = 'C:/Users/user/Dropbox/Course/gadurate courses/Machine Learning/hw6/correct_answer.csv'
correct_answer = pd.read_csv(correct_answer_path)
ca = np.array(correct_answer['ANS'])
sum(result-ca)/1980000
'''

"""


def meanX(dataX):
    return np.mean(dataX, axis=0)

def pca(XMat, k):
    average = meanX(XMat)
    m, n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T) 
    featValue, featVec = np.linalg.eig(covX)  
    index = np.argsort(-featValue)
    finalData = []
    if k > n:
        print("k must lower than feature number")
        return
    else:
        # 注意特征向量时列向量，而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
        selectVec = np.matrix(featVec.T[index[:k]])
        finalData = data_adjust * selectVec.T
        reconData = (finalData * selectVec) + average
    return finalData, reconData

finalData, reconData = pca(image, 64)

FinalData = np.array(finalData)
#FinalData = np.array(np.transpose(finalData))

sc = plt.scatter(np.array(np.transpose(finalData))[0], np.array(np.transpose(finalData))[1])
plt.show()

kmeans = KMeans(n_clusters=2, random_state=0).fit(FinalData)
evaluate_testing(kmeans.labels_)

for i in range(61,65):
    II = np.reshape(reconData[i], (28, 28))
    imgplot = plt.imshow(II)
    plt.show()

"""
