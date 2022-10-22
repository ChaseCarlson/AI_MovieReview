from flask import Flask, request, render_template
import predict

app = Flask(__name__)

@app.route('/', methods=["POST", "GET"])
def submit_review():
	if request.method == "POST":
		review = request.form["review"]
		sentiment, confidence = predict.predict_sentiment(review)
		color = "green" if sentiment == "Positive" else "red"
		return "<p style='color: " + color + ";'> The review was " + sentiment.lower() + "</p>"
	else:
		return render_template("review_form.html")
if __name__ == '__main__':
    app.run()