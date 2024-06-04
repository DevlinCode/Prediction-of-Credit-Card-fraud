import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
import streamlit as st
from sklearn.model_selection import RepeatedKFold
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# load data
data = pd.read_csv('creditcard.csv')

# Removing such data
df = data[data['Amount'] != 0]
df[data["Amount"]==0.00].head()

# split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]

# Lets perform RepeatedKFold
rkf = RepeatedKFold(n_splits=5,n_repeats=10,random_state=None)

# X is the feature set and y is the target
for train_index, test_index in rkf.split(X,y):
    print("Train:",train_index, "Test:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Oversample transactions to balance the classes
smote = SMOTE(random_state=0)
X_train, y_train = smote.fit_resample(X_train,y_train)
X_test, y_test = smote.fit_resample(X_test,y_test)


# train logistic regression model
model = xgb.XGBClassifier(learning_rate=0.2,max_depth=5,max_features='sqrt', min_sample_leaf=1,min_sample_split=2,n_estimators=150)
model.fit(X_train, y_train)


# evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)
print("Train Accuracy: ",train_acc,"Test Accuracy: ",test_acc)

# create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# create input fields for user to enter feature values
input_df = st.text_input('Input All features')
input_df_lst = input_df.split(',')
# create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    # get input feature values
    features = np.array(input_df_lst, dtype=np.float64)
    # make prediction
    prediction = model.predict(features.reshape(1,-1))
    # display result
    if prediction[0] == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")
