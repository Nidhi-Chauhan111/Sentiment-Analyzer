import string
import streamlit as st
import joblib

#loading model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("TFIDF_vectorizer.pkl")

def clean_text(text):
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = ' '.join(text.split())  # Remove extra whitespace 
    return text


st.title("Let us detect your sentiment")
st.subheader("Enter a product review to predict the sentiment")

#take input
user_input = st.text_area("Type your review here")

page_bg_img = '''
<style>
body {
background-image: url("https://www.istockphoto.com/photo/highly-detailed-graphic-material-for-your-projects-gm949779424-259253758?utm_campaign=adp_photos_sponsored&utm_content=https%3A%2F%2Funsplash.com%2Fphotos%2Fan-abstract-stain-on-a-pale-beige-background-CG2iuoTQ6LQ&utm_medium=affiliate&utm_source=unsplash&utm_term=art%3A%3A%3A");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)


with st.sidebar:
    st.header("About")
    st.markdown("This app predicts the sentiment of product reviews using a Naive Bayes classifier trained on real user feedback.")

#Prediction
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned_input = clean_text(user_input)  
        review_vector = vectorizer.transform([user_input])
        prediction = model.predict(review_vector)[0]
        
        st.success(f"Predicted Sentiment: **{prediction.upper()}**")
        if prediction == "positive":
            st.image("https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExOWoyeXV5OWZmdnRocXNobHR5M29xYnBzaDl0NDh1cnhraWM3ZmR2byZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3NtY188QaxDdC/giphy.gif", width=300)
            st.balloons()
        else:
            st.image("https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExNWdmdGJxcnZsMWhmODZmMjFpZ25pc3UxZngwcTJ2cDlvaHB4eXZiaSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/9Y5BbDSkSTiY8/giphy.gif", width=300)
            st.error("😢💔")

