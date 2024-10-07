# Import libraries and methods/functions
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

#Load CSV files
df1 = pd.read_csv('telecom_demographics.csv')
df2 = pd.read_csv('telecom_usage.csv')

#Merge the datasets
churn_df = pd.merge(df1, df2, on='customer_id', how='inner')

#Calculate and print churn rate
churn_rate = churn_df['churn'].mean()
churn_rate

#Identify and print categorical variables
categorical_vars = churn_df.select_dtypes(include=['object']).columns.tolist()
categorical_vars

#Convert Categorical variables to featured scale
#One Hot encoding
churn_df = pd.get_dummies(churn_df, columns=categorical_vars)

#Split features and target variable
#Drop columns that are not a feature
features = churn_df.drop(['customer_id', 'churn'], axis=1)
target = churn_df['churn']

#Feature scaling
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


#Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

#Logistic Regression model and Prediction
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)

#Random Forest and Prediction
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

#Logistic Regression evaluation
print("Logistic Regression Results :", confusion_matrix(y_test, logreg_pred))
print(classification_report(y_test, logreg_pred))

#RF evaluation
print("\nRandom Forest Results :", confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

#Accuracy scores
logreg_accuracy = accuracy_score(y_test, logreg_pred)
print("LogisticRegression Accuracy:", logreg_accuracy)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_accuracy)

#Determine the model with higher accuracy
higher_accuracy = 'LogisticRegression' if logreg_accuracy > rf_accuracy else 'RandomForest'
print(f"Model with Higher Accuracy: {higher_accuracy}")
