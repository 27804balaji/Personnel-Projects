import pandas as p
import numpy
import seaborn as s
import matplotlib.pyplot as pl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import string
import re

data_of_fake = p.read_csv(r"C:\Users\abc\Downloads\Fake.csv")
data_of_true = p.read_csv(r"C:\Users\abc\Downloads\True.csv")
# print(data_of_fake.head())
# print(data_of_true.head())
data_of_fake.rename(columns = {
                                'title' : 'Title',
                                'text' : 'Text',
                                'subject' : 'Subject',
                                'date' : 'Date'
                            } , inplace = True)

data_of_true.rename(columns = {
                                'title' : 'Title',
                                'text' : 'Text',
                                'subject' : 'Subject',
                                'date' : 'Date'
                            } , inplace = True)
# print(data_of_fake.head())
# print(data_of_true.head())

data_of_fake['Class'] = 0
data_of_true['Class'] = 1
# print(data_of_fake.head())
# print(data_of_true.head())
# print(data_of_fake.shape , data_of_true.shape)

fake_test = data_of_fake.tail(10)
for i in range(23480 , 23470 , -1):
    data_of_fake.drop([i] , axis = 0 , inplace=True)
    
true_test = data_of_true.tail(10)
for i in range(21416 , 21406 , -1):
    data_of_true.drop([i] , axis = 0 , inplace=True)

# print(data_of_fake.shape , data_of_true.shape)
fake_test['Class'] = 0
true_test['Class'] = 1
# print(fake_test.head(10))
# print(true_test.head(10))

merged_data = p.concat([data_of_fake , data_of_true] , axis = 0)
# print(merged_data.head(10))
# print(merged_data.isnull().sum())

def word(text):
    text = text.lower()
    text = re.sub('\[.*?\]' , '' , text)
    text = re.sub('\\W' , ' ' , text)
    text = re.sub('https?://\S+|www\.\S+' , '' , text)
    text = re.sub('<.*?>+' , '' , text)
    text = re.sub('[%s]' % re.escape(string.punctuation) , '' , text)
    text = re.sub('\n' , '' , text)
    text = re.sub('\w*\d\w*' , '' , text)
    return text


merged_data['ext'] = merged_data['Text'].apply(word)
data = merged_data.drop(['Title' , 'Text' , 'Subject'] , axis = 1)
data = data.sample(frac = 1) #It allows users to create fraction instances that can be created from a pair of integers(numerator and denominator), from a rational number, or even from a string
# print(data.head(10))

data.reset_index(inplace = True)
data.drop(['index'] , axis = 1 , inplace = True)
# print(data.head(10))

x = merged_data['Text']
y = merged_data['Class']

from sklearn.feature_extraction.text import TfidfVectorizer
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

vector = TfidfVectorizer()
xv_train = vector.fit_transform(x_train)
xv_test = vector.transform(x_test)

model = LogisticRegression()
fitting = model.fit(xv_train, y_train)
predictions = model.predict(xv_test)
print('The Prediction :', predictions)
print('The Prediction Accuracy :',accuracy_score(y_test, predictions))
print('The Classification Report :\n',classification_report(y_test, predictions))