import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

# Load dataset
df = pd.read_csv("AmesHousing.csv")

# Feature selection
features = ['Neighborhood', 'Overall Qual', 'Gr Liv Area', 'Garage Cars',
            'Total Bsmt SF', 'Year Built', 'Full Bath', 'Bedroom AbvGr']
target = 'SalePrice'
df = df[features + [target]].dropna()

# Encode categorical
le = LabelEncoder()
df['Neighborhood'] = le.fit_transform(df['Neighborhood'])

X = df[features]
y = df[target]

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save model and encoders
model.save("house_price_model.h5")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
