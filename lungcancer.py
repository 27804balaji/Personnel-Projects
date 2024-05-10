import pandas as p
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

data = p.read_csv(r"C:\Users\abc\Downloads\lung_cancer_survey (1).csv")
data.rename(columns = {
    'GENDER' : 'Gender',
    'AGE' : 'Age' , 
    'SMOKING' : 'Smoking' , 
    'YELLOW_FINGERS' : 'YellowFingerS' ,
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
# print(data.head())
data_1 = data[['Gender' , 'Age' , 'Smoking' , 'YellowFingerS' , 'Anxiety' , 'PeerPressure' , 'ChronicDisease' , 'Fatigue' , 'Allergy' , 'Wheezing' , 'ALcoholConsumption' , 'Coughing' , 'ShortnessOfBreadth' , 'SwallowingDifficulty' , 'ChestPain']] 
x_train , x_test , y_train , y_test = train_test_split(data_1 , data.LungCancer , test_size = 0.3)
# print(len(x_train))
# print(len(x_test))
# print(data.to_string())

# score_lr = cross_val_score(LogisticRegression() , data_1 , data.LungCancer)
# score_svc = cross_val_score(SVC() , data_1 , data.LungCancer)
# score_rf = cross_val_score(RandomForestClassifier(n_estimators=40 ,  max_features = 'sqrt') , data_1 , data.LungCancer)
# print('The Score of Logistic Regrssion' , score_lr)
# print('The Score of Support Vector Classifier' ,score_svc)
# print('The Score of Random Forest Classifier' , score_rf)

gender = int(input('Enter the Gender :'))
age= int(input('Enter the Age :'))
smoking = int(input('Enter the Smoking Level :'))
yellow = int(input('Enter Yellow Fingers Level :'))
anxiety = int(input('Enter the Anxiety Level :'))
peer = int(input('Enter the Peer Pressure Level :'))
chronic = int(input('Enter the Chronic Disease Level :'))
fatigue = int(input('Enter the Fatigue Level :'))
allergy = int(input('Enter the Allergy Level :'))
wheezing = int(input('Enter the Wheezing Level :'))
alchocol = int(input('Enter the Alcohol Consumption Level :'))
cough = int(input('Enter the Coughing Level :'))
breadth = int(input('Enter the Shortness of Breadth Level :'))
swallowing = int(input('Enter the Swallowing Difficulty Level :'))
chest = int(input('Enter the Chest Pain Level :'))

inputs = [[gender , age , smoking , yellow , anxiety , peer , chronic , fatigue , allergy , wheezing , alchocol , cough , breadth , swallowing , chest]]
model = RandomForestClassifier(n_estimators=40 , max_features = 'sqrt')
fiting = model.fit(x_train , y_train)
prediction = model.predict(inputs)
if prediction == 1:
    print('Cancer Positive(+ve)')
else :
    print('Cancer Negative(-ve)')

print('The Score :' ,model.score(x_test , y_test)*100,'%')
