from keras.models import Model
from keras.layers import Input, Dense,Lambda
from inceptionv3 import *
import os
import numpy as np
from keras.preprocessing import image
from keras.layers.merge import dot
import random
margin = 0.1

def tripletloss():
	q_input = Input(shape=(299,299,3))
	a_right = Input(shape=(299,299,3))
	a_wrong = Input(shape=(299,299,3))
	q_encoded = InceptionV3(q_input)
	a_right_encoded = InceptionV3(a_right)
	a_wrong_encoded = InceptionV3(a_wrong)

	right_cos = dot([q_encoded,a_right_encoded], -1, normalize=True)
	wrong_cos = dot([q_encoded,a_wrong_encoded], -1, normalize=True)
	
	loss = Lambda(lambda x: K.relu(margin+x[0]-x[1]))([wrong_cos,right_cos])
	
	model_train = Model(inputs=[q_input,a_right,a_wrong], outputs=loss)
	model_q_encoder = Model(inputs=q_input, outputs=q_encoded)
	model_a_encoder = Model(inputs=a_right, outputs=a_right_encoded)
	print(model_train.summary())
	return model_train,model_q_encoder,model_a_encoder
	


def eachFile(filepath):
	file = []
	pathDir =  os.listdir(filepath)
	for allDir in pathDir:
		child = os.path.join('%s/%s' % (filepath, allDir))
		file.append(child)
	return file

def getdata():
	folders = eachFile("flowers")
	print(folders)
	images = []
	labels = []
	for folder in folders:
		label = folder[-1]
		files = eachFile(folder)
		for file in files:
			img = image.load_img(file, target_size=(299, 299))
			input_image = image.img_to_array(img)
			input_image = preprocess_input(input_image)
			
			images.append(input_image)
			labels.append(label)

	labels = np.array(labels)
	qa_images = []
	
	qa_labels = []
	for i in range(len(images)):
		for j in range(i):
			if labels[i]==labels[j]:
				qa_images.append([images[i],images[j]])
				qa_labels.append(labels[i])


	
	qa_labels = np.array(qa_labels)
	q_input = []
	right_input = []
	wrong_input = []
	for i in range(len(qa_images)):
		wrong_indexs = np.where(labels!=qa_labels[i])[0]

		for j in wrong_indexs:
			q_input.append(qa_images[i][0])
			right_input.append(qa_images[i][1])
			wrong_input.append(images[j])

	
	print(len(q_input))
	q_input = np.array(q_input[:128])
	right_input = np.array(right_input[:128])
	wrong_input = np.array(wrong_input[:128])
	print(np.shape(q_input))
	return q_input,right_input,wrong_input
	
def train():
	model_train,model_q_encoder,model_a_encoder = tripletloss()
	model_train.compile(optimizer='adam', loss=lambda y_true,y_pred: y_pred)
	model_q_encoder.compile(optimizer='adam', loss='mse')
	model_a_encoder.compile(optimizer='adam', loss='mse')
	q,a1,a2 = getdata()
	y = np.random.randint(1,2,[len(q),1])
	model_train.fit([q,a1,a2], y, batch_size=1,epochs=1)	
	model_train.save("triplet.h5")


if __name__ == "__main__":
	train()
