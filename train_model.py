import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load the data we just created
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convert lists to NumPy arrays for Scikit-learn
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# 2. Split into Training (80%) and Testing (20%)
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# 3. Initialize and Train the Random Forest
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

# 4. Check Accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f'Model Training Complete! Accuracy: {score * 100:.2f}%')

# 5. Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
print("Model saved as 'model.p'")