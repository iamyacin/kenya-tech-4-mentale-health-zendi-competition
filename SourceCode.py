import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv('Data\Train.csv')
test = pd.read_csv('Data/Test.csv')

train = df.iloc[:, :-1]
target = df.iloc[:, -1]

labelencoder = LabelEncoder()
target = labelencoder.fit_transform(target)
# ALcohol = 0 --- Depression = 1 --- Drugs = 2 --- Suicide = 3 

text = pd.concat([train, test], ignore_index=True)

corpus = []
for i in range(0, 925):
    feeling = re.sub('[^a-zA-Z]', ' ', str(text['text'][i]))
    feeling = feeling.lower()
    feeling = feeling.split()
    ps = PorterStemmer()
    feeling = [ps.stem(word) for word in feeling if not word in set(stopwords.words('english'))]
    feeling = ' '.join(feeling)
    corpus.append(feeling)

vectorizer = TfidfVectorizer()
vector_of_words = vectorizer.fit_transform(corpus).toarray()

train_matrix = vector_of_words[:616,:]
test_matrix = vector_of_words[616:,:]

X_train, X_test, y_train, y_test = train_test_split(train_matrix, target, test_size = 0.20, random_state = 0)
    
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
        
y_pred = classifier.predict(X_test)
        
cv = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)


y_pred_proba_test_set = classifier.predict_proba(test_matrix)

Submission = test.iloc[:, 0].to_frame()
# ALcohol = 0 --- Depression = 1 --- Drugs = 2 --- Suicide = 3 
Submission['Depression']=y_pred_proba_test_set[:,1]
Submission['Alcohol']=y_pred_proba_test_set[:,0]
Submission['Suicide']=y_pred_proba_test_set[:,3]
Submission['Drugs']=y_pred_proba_test_set[:,2]

Submission.to_csv('Submissions/Submission7.csv', index=False)




