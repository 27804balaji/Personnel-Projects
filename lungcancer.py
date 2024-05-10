import pandas as p
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

data = p.read_csv(r"C:\Users\abc\Downloads\lung_cancer_survey (1).csv")
data.rename(columns = {
    'GENDER' : 'Gender',
    'AGE' : 'Age' , 
    'SMOKING' : 'Smoking' , 
    'ANXIETY' : 'Anxiety' , 
    'PEER_PRESSURE' : 'PeerPressure' , 
    'CHRONIC DISEASE' : 'ChronicDisease' , 
    'FATIGUE ' : 'Fatigue' , 
    'ALLERGY ' : 'Allergy' , 
    'WHEEZING' : 'Wheezing' ,
    'ALCOHOL CONSUMING' : 'ALcoholConsumption', 
    'COUGHING' : 'Coughing' , 
    'SHORTNESS OF BREATH' : 'ShortnessOfBreadth' ,
    'SWALLOWING DIFFICULTY' : 'SwallowingDifficulty' , 
    'CHEST PAIN' : 'ChestPain',
    'LUNG_CANCER' : 'LungCancer'
} , inplace = True)

label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data.Gender)
data['LungCancer'] = label_encoder.fit_transform(data.LungCancer)
data_1 = data[['Gender' , 'Age' , 'Smoking' , 'Anxiety' , 'PeerPressure' , 'ChronicDisease' , 'Fatigue' , 'Allergy' , 'Wheezing' , 'ALcoholConsumption' , 'Coughing' , 'ShortnessOfBreadth' , 'SwallowingDifficulty' , 'ChestPain']] 
x_train , x_test , y_train , y_test = train_test_split(data_1 , data.LungCancer , test_size = 0.3)
# print(len(x_train))
# print(len(x_test))
# print(data.to_string())

gender = int(input('Enter the Gender'))
age= int(input('Enter the Age'))
smoking = int(input('Enter the Smoking Level'))
anxiety = int(input('Enter the Anxiety Level'))
peer = int(input('Enter the Peer Pressure Level'))
chronic = int(input('Enter the Chronic Disease Level'))
fatigue = int(input('Enter the Fatigue Level'))
allergy = int(input('Enter the Allergy Level'))
wheezing = int(input('Enter the Wheezing Level'))
alchocol = int(input('Enter the Alcohol Consumption Level'))
cough = int(input('Enter the Coughing Level'))
breadth = int(input('Enter the Shortness of Breadth Level'))
swallowing = int(input('Enter the Swallowing Difficulty Level'))
chest = int(input('Enter the Chest Pain Level'))

inputs = [[gender , age , smoking , anxiety , peer , chronic , fatigue , allergy , wheezing , alchocol , cough , breadth , swallowing , chest]]

def get_score(model , x_train , x_test , y_train , y_test):
  model.fit(x_train , y_train)
  print(f'The Accuracy of the {model} :',model.score(x_test , y_test))

kf = KFold(n_splits = 3)
# for train_index , test_index in kf.split([1,2,3,4,5,6,7,8,9]):
#   print(train_index , test_index)

                                                              # Stratified K FoldKFold...
skf = StratifiedKFold(n_splits = 3)                                                               
x_train , x_test , y_train , y_test = train_test_split(inputs , data.LungCancer , test_size = 0.3) 
  

# Instead of the Above code we can use...
  # cross_val_score is a function which requires model , our x and y as the parameter...
print(cross_val_score(LogisticRegression() , inputs , data.LungCancer))
print(cross_val_score(SVC() , inputs , data.LungCancer))
print(cross_val_score(RandomForestClassifier(n_estimators = 40) , inputs , data.LungCancer))


