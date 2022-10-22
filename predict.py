import tensorflow as tf
from tensorflow import keras
import numpy as np
import util

data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)


model = keras.models.load_model("model_current.h5")

def predict_sentiment(review):
	test_review = util.encode_review(review)
	if len(test_review) < 1:
		return "Error"
	predict = model.predict([test_review])
	predict_score = float(predict[0])
	confidence = abs(0.5-predict_score)*2
	review_prediction_string = util.get_review_string(predict_score)
	return review_prediction_string, confidence