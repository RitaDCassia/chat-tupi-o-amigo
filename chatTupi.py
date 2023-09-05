import nltk
# Importa a biblioteca nltk para processar palavras
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
# Importa o lematizador do nltk para reduzir as palavras às suas formas base
lemmatizer = WordNetLemmatizer()

import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

# Lista de palavras
words = []
classes = []
documents = []
ignore_words = ['?', '!']
# Lê o arquivo JSON contendo as intenções e padrões de diálogo
data_file = open('tupi-o-amigo.json', encoding='utf-8').read()
intents = json.loads(data_file)

# Loop pelas intenções e padrões do arquivo JSON
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Processamento das palavras
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Criação dos arquivos pickle para armazenar as listas de palavras e classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Inicialização dos dados de treinamento
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
# Separação dos dados de treinamento em X (padrões) e Y (intenções)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Criação do modelo sequencial utilizando Keras
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilação do modelo com o otimizador SGD
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Treinamento do modelo
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Salvando o modelo treinado em um arquivo
model.save('chatTupi_model.h5')

print("Modelo criado")
import nltk
# Importa a biblioteca nltk para processar palavras
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
# Importa o lematizador do nltk para reduzir as palavras às suas formas base
lemmatizer = WordNetLemmatizer()

import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

# Lista de palavras
words = []
classes = []
documents = []
ignore_words = ['?', '!']
# Lê o arquivo JSON contendo as intenções e padrões de diálogo
data_file = open('tupi-o-amigo.json', encoding='utf-8').read()
intents = json.loads(data_file)

# Loop pelas intenções e padrões do arquivo JSON
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Processamento das palavras
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Criação dos arquivos pickle para armazenar as listas de palavras e classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Inicialização dos dados de treinamento
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
# Separação dos dados de treinamento em X (padrões) e Y (intenções)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Criação do modelo sequencial utilizando Keras
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilação do modelo com o otimizador SGD
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Treinamento do modelo
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Salvando o modelo treinado em um arquivo
model.save('chatTupi_model.h5')

print("Modelo criado")
