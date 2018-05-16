import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random
from lxml import html
import requests
from newsapi import NewsApiClient
import wikipedia
import pyowm
import webbrowser
from google_speech import Speech
# restore all of our data structures
import pickle


lang = "en"

data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intents file
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)

# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

#p = bow("is your shop open today?", words)
#print (p)
#print (classes)

# load our saved model
model.load('./model.tflearn')

# create a data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.25
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        val = random.choice(i['responses'])
                        speech = Speech(val, lang)
                        sox_effects = ("speed", "1.0")
                        speech.play(sox_effects)
                        # a random response from the intent
                        return print(val)

            results.pop(0)




while(1):
    ques = input("hey!! Ask me something -: ")
    print(classify(ques))
    response(ques)

    if 'wiki' in ques:
        z=ques
        print(wikipedia.summary(str(z),sentences=2))

    if'news' in ques:
        print('TOP NEWS : ')
        newsapi = NewsApiClient(api_key='439908f4580845f696e6a19fa868cfd5')
        top_headlines = newsapi.get_top_headlines(language='en')

        print(top_headlines)

    if 'google' in ques:
        new=2
        tabUrl="http://google.com/?#q="
        webbrowser.open(tabUrl+ques,new=new)

    if 'weather' in ques:
        owm=pyowm.OWM('7481da4f938822e5cf6bf62834c6d5da')
        a=input('enter city')
        observation=owm.weather_at_place(a)
        w=observation.get_weather()
        a=w.get_wind()
        print('wind- ',a)
        b=w.get_temperature('celsius')
        print('temp.- ',b)
        c=w.get_humidity()
        print('humidity- ',c)

    if ques == 'close':
        break
