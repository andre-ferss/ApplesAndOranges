# Importa bibliotecas necessárias
import pandas as pd  # Manipulação de dados (DataFrame)
from sklearn.model_selection import train_test_split  # Separar treino/teste
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Normalização e codificação
from sklearn.linear_model import LogisticRegression  # Modelo de regressão logística
from sklearn.neural_network import MLPClassifier # Modelo de MLP Classifier
import joblib  # Salvar arquivos

# Carrega o arquivo CSV contendo os dados de maçãs e laranjas
df = pd.read_csv("apples_and_oranges.csv")

# Separa as variáveis independentes (peso e tamanho) e a variável alvo (classe)
X = df[["Weight", "Size"]]  # Entradas
y = df["Class"]             # Saída (Apple/Orange)

# Codifica as classes (Apple e Orange em valores numéricos)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Padroniza os dados (média=0 e desvio padrão=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divide os dados em treino (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Cria e treina o modelo de Regressão Logística
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Cria e treina o modelo MLPClassifier
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp_model.fit(X_train, y_train)

# Salva o modelo, o scaler e o encoder treinados para uso posterior
joblib.dump(logistic_model, "logistic_model.pkl")
joblib.dump(mlp_model, "mlp_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "encoder.pkl")
