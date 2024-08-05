import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')



# Carregar dados (assumindo que os DataFrames train e test já estão carregados)
y = train['Survived']
features = ['Pclass', 'Sex', 'SibSp', 'Parch']
X = pd.get_dummies(train[features])

# Criar pipeline para escalonamento e treino do modelo
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', Perceptron(eta0=0.1, random_state=1))
])

# Usar cross-validation para avaliar o modelo
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f'Acurácia média na validação cruzada: {cv_scores.mean():.2f}')

# Treinar o modelo no conjunto de treino completo
pipeline.fit(X, y)

# Preparar dados de teste e fazer previsões para envio ao Kaggle
X_test = pd.get_dummies(test[features])
X_test = X_test.reindex(columns=X.columns, fill_value=0)  # Garantir que as colunas correspondam
y_test_pred = pipeline.predict(X_test)



