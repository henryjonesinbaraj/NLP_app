import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import pickle
from PIL import Image

import panel as pn
from sklearn.feature_extraction.text import TfidfVectorizer

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Function for sentiment prediction
def predict_sentiment(review):
    review = preprocessing(review)
    x = vectorizer.transform([review])
    prediction = model.predict(x)[0]
    return prediction

# Text preprocessing function
def preprocessing(text):
    # Your text preprocessing steps go here
    # Example: convert to lowercase
    return text.lower()

# Define the GUI components
happy_image_path = "/Users/henryjones/Desktop/nlp_code/Happy.jpeg"  # Replace with the actual path
angry_image_path = "/Users/henryjones/Desktop/nlp_code/angryface.jpeg"  # Replace with the actual path
topic_heading = pn.pane.Markdown("# Sentiment Analysis App", style={"font-size": "24px"})
review_input = pn.widgets.TextInput(placeholder="Type your review here", width=400)
submit_button = pn.widgets.Button(name="Submit", button_type="primary", width=100)
prediction_output = pn.Row()

# Load the images
happy_image = Image.open(happy_image_path)
angry_image = Image.open(angry_image_path)

# Define the callback function to update the prediction output
def update_prediction(event):
    review = review_input.value
    prediction = predict_sentiment(review)

    if prediction == 1:
        prediction_output.clear()
        prediction_output.append(pn.pane.PNG(happy_image, width=150, height=150))
    else:
        prediction_output.clear()
        prediction_output.append(pn.pane.PNG(angry_image, width=150, height=150))     
        

# Attach the callback function to the review input widget
submit_button.on_click(update_prediction)

# Create the Panel app
app = pn.Column(
    topic_heading,
    review_input,
    submit_button,
    prediction_output,
    width=600,
)

# Show the app
app.servable()