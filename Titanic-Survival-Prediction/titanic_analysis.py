import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("dataset/train.csv")

print("Missing values before handling:")
print(df.isnull().sum())

if 'Cabin' in df.columns:
    df.drop(columns=['Cabin'], inplace=True)

df = df.copy()
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

le = LabelEncoder()
df['Embarked'] = le.fit_transform(df['Embarked'])

print("\nDataset after encoding:")
print(df.head())

df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nDataset Split:")
print(f"Training Set: {X_train.shape}, Testing Set: {X_test.shape}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nDataset Split:")
print(f"Training Set: {X_train.shape}, Testing Set: {X_test.shape}")
