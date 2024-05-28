import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load dataset
data = pd.read_csv('spotify_data.csv')

# Feature Engineering
data['is_repeated'] = data['play_count'] > 1

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Extract temporal features
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek

# Select features
features = ['user_id', 'song_id', 'genre', 'artist', 'hour', 'day_of_week']
X = data[features]
y = data['is_repeated']

# Convert categorical variables to dummy/indicator variables
X = pd.get_dummies(X, columns=['user_id', 'song_id', 'genre', 'artist', 'hour', 'day_of_week'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")

# Save the model
joblib.dump(model, 'spotify_recommendation_model.pkl')