#  Product Review Sentiment Analysis

This is a simple Machine Learning project that predicts whether a given product review is **positive** or **negative** using a Naive Bayes model. The application is built with **Python**, uses **NLP (Natural Language Processing)** techniques, and is deployed via a **Streamlit** web app.

---

##  Features

- Input product reviews in a text box
- Predicts sentiment: **Positive** or **Negative**
- Fun visuals with emoji and GIF feedback
- Built using:
  - `sklearn` (Multinomial Naive Bayes)
  - `NLTK` (for text preprocessing)
  - `Streamlit` (for GUI)
  - `Pandas`, `joblib`, etc.

 ## Project Structure
 - app.py  # fronted using streamlit
 - product_reviews_final_data.csv  # dataset used(worth 2565 rows; synthetic data*)
 - Project_ProdReview_Sentiment_Analysis.ipnyb # colab notebook
 - sentiment_model.pkl  # Naive Bayes trained model
 - TFIDF_vectorizer.pkl # TFIDF vectorizer
 - README.md # project overview

## How to run:
-Install requirements: manually install
 - pip install streamlit scikit-learn pandas nltk joblib

-Launch the app
 - streamlit run app.py
 -Then visit the link shown in terminal (usually http://localhost:8501/).

## How it works:
1. Review text is preprocessed (stopword removal, stemming/lemmatization).

2. Text is vectorized using TF-IDF.

3. The model uses Multinomial Naive Bayes to predict sentiment.

4. The result is displayed with a happy or sad visual.

 ## Notes
- This model is trained on synthetic + cleaned Amazon-style product reviews.

- May not generalize well to highly domain-specific or sarcastic reviews.

- You can retrain the model with a larger, more diverse dataset.

## Deployment
- You can also deploy this app on:

1. Streamlit Cloud (streamlit.io)

2. Hugging Face Spaces

3. Render / Heroku / Vercel (with tweaks)

## Author
- Nidhi Singh Chauhan