# -*- coding: utf-8 -*-


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import streamlit as st
import json
import random
from PIL import Image
from keras.models import load_model


#pickle_in = open(r"C:\Users\Admin\Desktop\CHATBOT\chatbot_model_new_pk.pkl", 'rb') 
#classifier = pickle.load(pickle_in)

#Load the pkl model file
model1 = load_model(r'C:\Users\Admin\Desktop\CHATBOT\chatbot_model_new.h5')
#Load the dataset file
intents = json.loads(open(r'C:\Users\Admin\Desktop\CHATBOT\intents.json').read())
#Load the words file
words = pickle.load(open(r'C:\Users\Admin\Desktop\CHATBOT\words_1.pkl','rb'))
#Load the classes file
classes = pickle.load(open(r'C:\Users\Admin\Desktop\CHATBOT\classes_1.pkl','rb'))

#Text Data preprocessing
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                #if show_details:
                    #print ("found in bag: %s" % w)
    return(np.array(bag))


def predict_class(sentence, model1):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model1.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json,text):
    tag = ints[0]['intent']
    #result=''
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    
    return result

def chatbot_response(text):
    text=str(text)
    ints = predict_class(text, model1)
    res = getResponse(ints, intents,text)
    #res1=intents['res']
    return res

res3=chatbot_response('hello')

df = st.sidebar.selectbox("Navigation",["Chatbot","Text_Input"])

if df == 'Chatbot':
    st.title("Chatbot")
    image = Image.open(r"C:\Users\Admin\Desktop\CHATBOT\chatbot12.jpg")
    st.image(image,width=200)

    str1=st.text_input('Text_message')
    #res4 = chatbot_response(str1)
    
    if st.button("submit"):
        res4 = chatbot_response(str1)
        st.success(res4)

if df == 'Text_Input':
    st.text_input('Text_message')
        
        
    

