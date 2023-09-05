import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# Importa o lematizador do nltk para reduzir as palavras às suas formas base

import pickle
import numpy as np

from keras.models import load_model
# Importa a função para carregar um modelo pré-treinado
model = load_model('chatTupi_model.h5')
# Carrega o modelo treinado a partir de um arquivo

import json
import random
# Carrega a biblioteca json para lidar com arquivos JSON

# Carrega o arquivo JSON contendo as intenções e padrões de diálogo
intents = json.loads(open('tupi-o-amigo.json', encoding='utf-8').read())
# Carrega as palavras e classes pré-processadas a partir dos arquivos pkl
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

# Função para limpar a sentença, tokenizar e lematizar as palavras
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Função para criar uma matriz bag-of-words para uma sentença
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

# Função para prever a classe de uma sentença usando o modelo
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Função para obter a resposta adequada com base nas intenções e respostas no arquivo JSON
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
        else:
            result = "You must ask the right questions"
    return result

# Função principal que utiliza as funções anteriores para responder a uma mensagem
def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res
