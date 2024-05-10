import pandas as p
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import math

data = p.read_csv(r"C:\Users\abc\Downloads\titanic.csv")
# print(data)
independent_variable = data.drop(['Survived'] , axis = 'columns')
dependent_variable = data.Survived
# print(dependent_variable)
# print(independent_variable)

le_PassengerId = LabelEncoder()
le_Pclass = LabelEncoder()
le_Name = LabelEncoder()
le_Sex = LabelEncoder()
le_Ticket  = LabelEncoder()
le_Cabin = LabelEncoder()
le_Embarked  = LabelEncoder()

independent_variable['PassengerId_n'] = le_PassengerId.fit_transform(independent_variable['PassengerId'])
independent_variable['Pclass_n'] = le_Pclass.fit_transform(independent_variable['Pclass'])
independent_variable['Name_n'] = le_Name.fit_transform(independent_variable['Name'])
independent_variable['Sex_n'] = le_Sex.fit_transform(independent_variable['Sex'])
independent_variable['Ticket_n'] = le_Ticket.fit_transform(independent_variable['Ticket'])
independent_variable['Cabin_n'] = le_Cabin.fit_transform(independent_variable['Cabin'])
independent_variable['Embarked_n'] = le_Embarked.fit_transform(independent_variable['Embarked'])
# print(independent_variable.head())

independent_variable_1 = independent_variable.drop(['PassengerId' , 'Pclass' , 'Name' , 'Name_n', 'Sex' , 'Ticket' , 'Cabin' , 'Embarked'] , axis = 'columns')
print(independent_variable_1.head(5))

median_Age = math.floor(independent_variable_1.Age.median())
# print(median_Age)
independent_variable_1.Age = independent_variable_1.Age.fillna(median_Age)

id = int(input('Enter the PassengerId :'))
pclass = int(input('Enter the PCLass :'))
sex = int(input('Enter the Sex :'))
age = int(input('Enter the Age :'))
sibsp = int(input('Enter the SibSp :'))
parch = int(input('Enter the Parch :'))
ticket = int(input('Enter the Ticket :'))
fare = float(input('Enter the Fare :'))
cabin = int(input('Enter the Cabin :'))
embarked = int(input('Enter the Embarked :'))

inputs = [[id , pclass , sex , age , sibsp , parch , ticket , fare , cabin , embarked]]
model = tree.DecisionTreeClassifier()
fitting = model.fit(independent_variable_1 , dependent_variable)
prediction = model.predict(inputs)

if 0 in prediction:
  print('The Person may not be survived...')

elif 1 in prediction:
  print('The Person may be survived...')

score = model.score(independent_variable_1 , dependent_variable)
print('The Accuracy of the MOdel :' , score)