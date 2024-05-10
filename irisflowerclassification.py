import pandas as p
from sklearn.datasets import load_iris #load_iris means loading the iris flower data details...
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as pl

iris = load_iris()
# print(iris.feature_names)

data = p.DataFrame(iris.data , columns = iris.feature_names)
# Adding target column...
data['target'] = iris.target
#print(data.head())

data_1 = data['sepal length'] = data['sepal length (cm)']
data_2 = data['sepal width'] = data['sepal width (cm)']
data_3 = data['petal length'] = data['petal length (cm)']
data_4 = data['petal width'] = data['petal width (cm)']

# print(data_1 , data_2 , data_3 , data_4)
merge = p.concat([data ,data_1 , data_2 , data_3 , data_4 ] , axis = 'columns')
# print(merge.head())

drop = merge.drop(['sepal length (cm)' , 'sepal width (cm)' ,  'petal length (cm)' ,  'petal width (cm)'] ,  axis = 'columns') #drop is our new dataset...


                                                                #The Target contains three iris flower data details...
#target_name = iris.target_names
#print(target_name)
                                                                #To identify the type of flower...
# drop[drop.target == 0].head()
# drop[drop.target == 1].head()
# drop[drop.target == 2].head()
# print(drop)

# Applying the values of target to Flower_name...
drop['Flower_name'] = drop.target.apply( lambda x : iris.target_names[x])
# print(drop.head())

                                                                # Creating three different dataframes for three different flowers...
dataframe_1 = drop[drop.target == 0].head()
dataframe_2 = drop[drop.target == 1].head()
dataframe_3 = drop[drop.target == 2].head()

# pl.xlabel('Sepal length (cm)')
# pl.ylabel('Sepal width (cm)')
# plot_1 = pl.scatter(dataframe_1['sepal length'] , dataframe_1['sepal width'] , color = 'red' , marker = '.')
# plot_2 = pl.scatter(dataframe_2['sepal length'] , dataframe_2['sepal width'] , color = 'green' , marker = '.')
# print(plot_1)
# print(plot_2)

# pl.xlabel('Petal length (cm)')
# plot_3 = pl.ylabel('Petal width (cm)')
# plot_4 = pl.scatter(dataframe_1['petal length'] , dataframe_1['petal width'] , color = 'red' , marker = '.')
# pl.scatter(dataframe_2['petal length'] , dataframe_2['petal width'] , color = 'green' , marker = '.')
# print(plot_3)
# print(plot_4)

id_variable = drop.drop(['target' , 'Flower_name'] , axis = 'columns')
d_variable = drop.target
# print(id_variable)
# print(d_variable)

x_train , x_test , y_train , y_test = raining_data = train_test_split(id_variable , d_variable , test_size = 0.2)
# print(len(x_train) , len(x_test))
model = SVC()

fitting = model.fit(x_train , y_train)
score_1 = model.score(x_train , y_train)
print('The Accuracy of Model Learning :',score_1)
prediction = model.predict(x_test)
print('The Prediction :',prediction)
score_2 = model.score(x_test , y_test)
print('The Accuracy of Model Prediction :' , score_2)
