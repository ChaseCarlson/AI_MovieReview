import predict

while True:
	review = input("Review: ")
	print("The review is", predict.predict_sentiment(review).lower())