import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error

#Loading data
data = fetch_california_housing()
X = data.data
Y = data.target

#Particion de Entrenamiento, Validacion y prueba
x_train, x_rest, y_train , y_rest = train_test_split(X,Y,test_size=0.4,random_state=321)
x_test , x_val , y_test, y_val = train_test_split(X,Y,test_size=0.5,random_state=123)


#Escalamieinto
scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
# x_val_s = scaler.transform(x_val)
x_test_s = scaler.transform(x_test)


#creacion, entrenamiento y validacion de los modelos

#Regrecion linar Modelo
ln = LinearRegression()
ln.fit(x_train_s,y_train)
y_pred_ln = ln.predict(x_test_s)
print("Regrecion Lineal - RMSE: ", np.sqrt(mean_squared_error(y_test,y_pred_ln)))


#Regrecion Logistica

y_class = (Y > np.median(Y)).astype(int)
y_train_class = (y_train > np.median(Y)).astype(int)
y_test_class = (y_test > np.median(Y)).astype(int)

log_reg = LogisticRegression()
log_reg.fit(x_train_s,y_train_class)
y_pred_log_reg = log_reg.predict(x_test_s)
print("Regrecion Logistica - Accuracy: ", accuracy_score(y_test_class, y_pred_log_reg))


#Arboles de Deciciones

tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train,y_train)
y_pred_tree_reg = tree_reg.predict(x_test)
print("Arbol de Desiciones - MAE: ", mean_absolute_error(y_test, y_pred_tree_reg))



# print("Colimnas", data.feature_names)
# print("x shape", X.shape)
# print("Y shape", Y.shape)