import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime

data = pd.read_csv('car.csv')

data_x = data.iloc[:,[1,3,4,5,6,7,8]]
data_y = data.iloc[:,2]

x= datetime.datetime.now()

fuel_type={"CNG" : 1,"Diesel" : 2, "Petrol" : 3}
Transmission={"Automatic" : 1, "Manual" : 2}
Seller_Type={"Dealer" : 1, "Individual" : 2}




data_x=data_x.replace({"Fuel_Type" : fuel_type, "Transmission" : Transmission,"Seller_Type": Seller_Type})
data_x=data_x.values
data_x[:,0]=x.year-data_x[:,0]


X_train,X_test,y_train,y_test=train_test_split(data_x,data_y,test_size=0.3,random_state=0)

reg = LinearRegression()
reg.fit(X_train,y_train)

print("Train Score: ",reg.score(X_train,y_train))
print("Test Score : ",reg.score(X_test,y_test))

pickle.dump(reg, open('car.pkl','wb'))

