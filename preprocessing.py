import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('../data/creditcard.csv')

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Save the preprocessed data
pd.DataFrame(X_train).to_csv('../data/X_train.csv', index=False)
pd.DataFrame(X_test).to_csv('../data/X_test.csv', index=False)
pd.DataFrame(y_train).to_csv('../data/y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('../data/y_test.csv', index=False)
