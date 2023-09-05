import streamlit as st
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from sklearn.feature_extraction.text import CountVectorizer

st.title("Email/SMS Spam Classifier")
ps = PorterStemmer()

nltk.download('punkt')
nltk.download('wordcloud')


def transform_text(text):
    #lowercase
    text = text.lower()
    #word tokenize
    text = nltk.word_tokenize(text)
    #remove special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    #stop punctuation
    text = list(y)
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    #stemming
    text = list(y)
    y.clear()
    for i in text:
        y.append(ps.stem(i))      
    return ' '.join(y)
model = pickle.load(open(r"datasets/smsmodel.txt","rb"))
count = pickle.load(open(r"datasets/countvectorsms.txt","rb"))
input_sms = st.text_area("Enter the message")
if st.button('Predict'):
    #preprocess
    transformed_text = transform_text(input_sms)
    #vectorize
    vector_input = count.transform([transformed_text])
    #predict
    result = model.predict(vector_input)[0]
    #result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")




        
