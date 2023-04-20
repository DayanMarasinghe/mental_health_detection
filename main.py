# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Load the datasets
dep_data = pd.read_csv('depression_dataset.csv')
anx_data = pd.read_csv('anxiety_dataset.csv')
str_data = pd.read_csv('stress_dataset.csv')

# Combine the datasets into a single dataframe
data = pd.DataFrame({'text': dep_data['text'].tolist() + anx_data['text'].tolist() + str_data['text'].tolist(),
                     'depression_label': dep_data['label'].tolist() + [0] * len(anx_data) + [0] * len(str_data),
                     'anxiety_label': [0] * len(dep_data) + anx_data['label'].tolist() + [0] * len(str_data),
                     'stress_label': [0] * len(dep_data) + [0] * len(anx_data) + str_data['label'].tolist()})

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
train_features = vectorizer.fit_transform(train_data['text'])
test_features = vectorizer.transform(test_data['text'])

# Train a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(train_features, train_data[['depression_label', 'anxiety_label', 'stress_label']])

# Make predictions on the testing set
pred_labels = clf.predict(test_features)

# Evaluate the performance of the model
print('Accuracy:', accuracy_score(test_data[['depression_label', 'anxiety_label', 'stress_label']], pred_labels))
print('Precision:',
      precision_score(test_data[['depression_label', 'anxiety_label', 'stress_label']], pred_labels, average='micro'))
print('Recall:',
      recall_score(test_data[['depression_label', 'anxiety_label', 'stress_label']], pred_labels, average='micro'))
print('F1-score:',
      f1_score(test_data[['depression_label', 'anxiety_label', 'stress_label']], pred_labels, average='micro'))

# Make predictions on new text data
new_text = ['I feel very sad and hopeless', 'I am constantly worried about everything',
            'I am under a lot of pressure at work', 'I hate my life and want to kill myself',
            'I have a sad life filled with sorrow']
new_features = vectorizer.transform(new_text)
new_pred_labels = clf.predict(new_features)

# specify the column names
column_names = ["Depression", "Anxiety", "Stress"]

column_sums = np.sum(new_pred_labels, axis=0)

# iterate through the column names and sums and print them
for i, column_sum in enumerate(column_sums):
    print(f"{column_names[i]}: {column_sum}")

# Find the index of the maximum value in column_sums
max_index = np.argmax(column_sums)

# Print the column name with the highest sum
print("Final label: " + column_names[max_index])
