import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Criando um DataFrame fictício
data = {
    'Feature1': [2.0, 3.0, 5.0, 7.0, 10.0],
    'Feature2': [15.0, 20.0, 25.0, 30.0, 35.0]
}

df = pd.DataFrame(data)

print("DataFrame original:")
print(df)

# Utilizando StandardScaler do scikit-learn
scaler = StandardScaler()
scaled_data_sklearn = scaler.fit_transform(df)

print("\nPadronização usando scikit-learn:")
print(scaled_data_sklearn)

# Utilizando a padronização do pandas
scaled_data_pandas = (df - df.mean()) / df.std()

print("\nPadronização usando pandas:")
print(scaled_data_pandas)