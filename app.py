from flask import Flask, render_template, jsonify, request
import processor

# Cria uma instância da classe Flask
app = Flask(__name__)

# Define uma chave secreta para a aplicação Flask
app.config['SECRET_KEY'] = 'solved123'

# Define a rota principal para a página inicial
@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())

# Define a rota para a resposta do chatbot
@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():
    if request.method == 'POST':
        # Obtém a pergunta enviada pelo usuário via formulário
        the_question = request.form['question']

        # Chama a função do processador para obter a resposta do chatbot
        response = processor.chatbot_response(the_question)

    # Retorna a resposta do chatbot em formato JSON
    return jsonify({"response": response })

# Inicia a aplicação Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8888', debug=True)
