# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:52:58 2022

@author: Admin
"""

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open(r'C:\Users\Admin\Desktop\CHATBOT\intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words_1 = sorted(list(set(words)))

# sort classes
classes_1 = sorted(list(set(classes)))

# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)

pickle.dump(words,open(r'C:\Users\Admin\Desktop\CHATBOT\words_1.pkl','wb'))
pickle.dump(classes,open(r'C:\Users\Admin\Desktop\CHATBOT\classes_1.pkl','wb'))


# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
output_empty
documents
classes
# training set, bag of words for each sentence
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        if w in pattern_words:
            bag.append(1) 
        else:
            bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array

random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
len(train_x)
len(train_y)
print("Training data created")



len(train_x[0])
#Model building
# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
#For output purpose
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist=model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save(r'C:\Users\Admin\Desktop\CHATBOT\chatbot_model_new.h5', hist)

#pickle_out = open(r"C:\Users\Admin\Desktop\CHATBOT\chatbot_model_new_pk.pkl", mode = "wb") 
#pickle.dump(model, pickle_out) 
#pickle_out.close()

model.summary()
type(train_x[0])
model.predict(np.array([train_x[0]]))[0]
print(np.array(train_y[0]))
accuracy = model.evaluate(train_x, train_y)
accuracy
print('Accuracy: %.2f' % (accuracy[1]*100))
train_x[0]

results1=[(0,0.12),(1,0.55),(2,0.35)]
ty=results1.sort(key=lambda x: x[1], reverse=True)

