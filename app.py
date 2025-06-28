import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Cargar CSV
df = pd.read_csv('data.csv')

# Mostrar primeras filas
print("📄 Mostrando los primeros registros del dataset:")
print(df.head())

# ----------------------------
# 🔄 Preprocesamiento de datos
# ----------------------------

# Convertir fecha
df['Factura_Fecha'] = pd.to_datetime(df['Factura_Fecha'], errors='coerce')
df['Factura_Mes'] = df['Factura_Fecha'].dt.month.fillna(0).astype(int)
df['Factura_DiaSemana'] = df['Factura_Fecha'].dt.dayofweek.fillna(0).astype(int)

# Codificar variables categóricas
le_cliente = LabelEncoder()
le_articulo = LabelEncoder()
le_tipopersona = LabelEncoder()
le_categoria = LabelEncoder()
le_sucursal = LabelEncoder()

df['Cliente_Id_enc'] = le_cliente.fit_transform(df['Cliente_Id'].fillna('Unknown'))
df['Articulo_Id_enc'] = le_articulo.fit_transform(df['Articulo_Id'].fillna('Unknown'))
df['TipoPersona_enc'] = le_tipopersona.fit_transform(df['TipoPersona'].fillna('Unknown'))
df['Categoria_Id_enc'] = le_categoria.fit_transform(df['Categoria_Id'].fillna('Unknown'))
df['Suc_Id_enc'] = le_sucursal.fit_transform(df['Suc_Id'].fillna('Unknown'))

# Asegurar que la cantidad sea numérica
df['Detalle_Cantidad'] = pd.to_numeric(df['Detalle_Cantidad'], errors='coerce').fillna(0)

# ----------------------------
# 📊 Construcción de features y target
# ----------------------------

# Variables de entrada (X)
X = df[['Cliente_Id_enc', 'TipoPersona_enc', 'Categoria_Id_enc',
        'Detalle_Cantidad', 'Factura_Mes', 'Factura_DiaSemana', 'Suc_Id_enc']]

# Variable objetivo (y)
y = df['Articulo_Id_enc']

# Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Número de clases
num_classes = len(le_articulo.classes_)

# One-hot encoding para el target
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# ----------------------------
# 🧠 Definir modelo
# ----------------------------
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ----------------------------
# 🚀 Entrenar el modelo
# ----------------------------
print("🔄 Entrenando el modelo... esto puede tardar unos minutos.")
model.fit(X_train, y_train_cat, epochs=30, batch_size=32, validation_data=(X_test, y_test_cat))

# ----------------------------
# ✅ Evaluar el modelo
# ----------------------------
print("✅ Entrenamiento completado. Evaluando el modelo...")
loss, acc = model.evaluate(X_test, y_test_cat)
print(f'🎯 Precisión en test: {acc:.4f}')

# ----------------------------
# 💾 Guardar el modelo
# ----------------------------
model.save("modelo_productos.h5")
print("💾 Modelo guardado como 'modelo_productos.h5'")
