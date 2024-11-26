import random
import json
import pickle
import numpy as np

import nltk
# nltk.download('punkt_tab')
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer # worked, working --> work

import tensorflow as tf

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json", encoding='utf-8').read())

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

model = tf.keras.models.load_model("chatbot_model.keras")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# converting word into bag of words, i.g 0 and 1
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words) 
    
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    
    return return_list

def get_response(intents_list, intents_json):
    try:
        tag = intents_list[0]["intent"]
    except:
        tag = ''

    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
        else:
            result = 'К сожалению, не могу ответить на Ваш вопрос'
    return result

while True:
    query = input("> ")
    ints = predict_class(query)
    res = get_response(ints, intents)
    print(res)
