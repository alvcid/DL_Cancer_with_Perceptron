from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Cargar datos
breast_cancer = load_breast_cancer()

X = breast_cancer.data
Y = breast_cancer.target

# print(dir(breast_cancer))

# print(X)

#Visualizar los datos
df = pd.DataFrame(X, columns=breast_cancer.feature_names)
# print(df)

# print(Y)

# Dividir conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(df, Y, stratify=Y)

print(f"Tamaño del conjunto de datos de entrenamiento: {len(X_train)}")
print(f"Tamaño del conjunto de datos de prueba: {len(X_test)}")

# Implementar la neurona MPNeuron más avanzada (Perceptrón)
class MPNeuron:
    def __init__(self):
        self.threshold = None

    def model(self, x):
        return (sum(x) >= self.threshold)

    def predict(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)

    def fit(self, X, Y):
        accuracy = {}
        # Seleccionamos un threshold entre el número de características de entrada
        for th in range(X.shape[1] + 1):
            self.threshold = th
            Y_pred = self.predict(X)
            accuracy[th] = accuracy_score(Y_pred, Y)
        # Seleccionamos el threshold que mejores resultados proporciona
        self.threshold = max(accuracy, key=accuracy.get)

# Transformar características de entrada a un valor binario
X_train_bin = X_train.apply(pd.cut, bins=2, labels=[0, 1])
X_test_bin = X_test.apply(pd.cut, bins=2, labels=[0, 1])

# print(X_test_bin)

# Instanciamos el modelo
mp_neuron = MPNeuron()

# Threshold óptimo
mp_neuron.fit(X_train_bin.to_numpy(), y_train)

print(f"Threshold óptimo: {mp_neuron.threshold}")

# Realizamos predicciones para ejemplos nuevos que no ha visto el modelo
Y_pred = mp_neuron.predict(X_test_bin.to_numpy())

print(Y_pred)

# Calculamos la exactitud de nuestra predicción
print(accuracy_score(y_test, Y_pred))

# Calculamos la matriz de confusión
print(confusion_matrix(y_test, Y_pred))
