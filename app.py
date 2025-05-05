# Importa as bibliotecas necessárias
from flask import Flask, request, jsonify, render_template  # Flask para API web
import joblib  # Para carregar os modelos treinados
import numpy as np  # Para manipulação de arrays numéricos

# Cria a aplicação Flask
app = Flask(__name__)

# Carrega o modelo treinado, o scaler e o encoder do disco
scaler = joblib.load('scaler.pkl')    # Padronizador de dados
encoder = joblib.load('encoder.pkl')  # Codificador das classes (Apple/Orange)

# Rota principal que renderiza a página inicial (index.html)
@app.route('/')
def home():
    return render_template('index.html')  # Exibe o HTML para o usuário

# Rota que recebe dados para predição
@app.route('/predict', methods=['POST'])
def predict():
    # Recebe os dados enviados pelo frontend
    data = request.get_json()
    weight = data['weight']
    size = data['size']
    model_type = data['model_type']  # Tipo de modelo (logistic ou mlp)

    # Carrega o modelo apropriado
    if model_type == 'logistic':
        print("Usando modelo: Regressão Logística")
        model = joblib.load("logistic_model.pkl")
    elif model_type == 'mlp':
        print("Usando modelo: MLP Classifier")
        model = joblib.load("mlp_model.pkl")
    else:
        return jsonify({'error': 'Tipo de modelo inválido'}), 400

    # Prepara os dados para predição
    input_data = np.array([[weight, size]])
    input_scaled = scaler.transform(input_data)

    # Faz a predição
    prediction = model.predict(input_scaled)
    result = encoder.inverse_transform(prediction)[0]  # Converte para Apple/Orange

    return jsonify({'result': result})

# Executa o servidor Flask em modo de depuração
if __name__ == '__main__':
    app.run(debug=True)
