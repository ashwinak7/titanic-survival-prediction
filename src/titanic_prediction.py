import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
train = pd.read_csv(r'C:\Users\RAJESH\Desktop\ML\Scikit-Learn\Dataset\Titanic\train.csv')
test = pd.read_csv(r'C:\Users\RAJESH\Desktop\ML\Scikit-Learn\Dataset\Titanic\test.csv')

# Preprocess training data
train = train.drop(columns=['Name', 'Embarked', 'Ticket', 'Cabin'])
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
train = train.dropna()

# Plot 1: Survival Count with counts on bars
plt.figure(figsize=(6,4))
ax = sns.countplot(x='Survived', data=train)
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom')
plt.show()

# Plot 2: Survival by Gender with counts on bars
plt.figure(figsize=(6,4))
ax = sns.countplot(x='Sex', hue='Survived', data=train)
plt.title('Survival by Gender')
plt.xlabel('Sex (0 = Male, 1 = Female)')
plt.ylabel('Count')
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom')
plt.show()

# Prepare features and target
X = train.drop(columns='Survived')
y = train['Survived']

# Split into training and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)
y_probs = model.predict_proba(X_val)[:, 1]

# Evaluate performance
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("ROC AUC Score:", roc_auc_score(y_val, y_probs))

# Plot 3: Confusion Matrix heatmap
plt.figure(figsize=(5,4))
cm = confusion_matrix(y_val, y_pred)
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot 4: ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_probs)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()

# Preprocess test data (for real-time predictions, no true labels)
test = test.drop(columns=['Name', 'Embarked', 'Ticket', 'Cabin'])
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
test = test.fillna(test.mean())

# Predict on test set
test_preds = model.predict(test)
print("Test Predictions (first 10):", test_preds[:10])

# After preprocessing the train data (dropping columns, mapping 'Sex', dropping NaNs)
train.to_csv(r'C:\Users\RAJESH\Desktop\ML\Scikit-Learn\Dataset\Titanic\train_preprocessed.csv', index=False)

# After preprocessing the test data (dropping columns, mapping 'Sex', filling NaNs)
test.to_csv(r'C:\Users\RAJESH\Desktop\ML\Scikit-Learn\Dataset\Titanic\test_preprocessed.csv', index=False)
