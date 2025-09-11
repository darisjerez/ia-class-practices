import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


dataset = pd.read_csv("./Datasets/dataset_spam_limpio.csv")

#asignando feature y labels
vectorice = CountVectorizer()
x = vectorice.fit_transform(dataset['mensaje'])
y = dataset['spam']

#Split de los datos de entresamiento y test
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.2, random_state=321)

#Entrenando modelos 

mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_pred_mbn = mnb.predict(x_test)
print("Resultado: ", accuracy_score(y_test, y_pred_mbn))


#Funcion para probar el dectector del spam

def spamDetector(mesanje):
    mensajeV = vectorice.transform([mesanje])
    prediccion = mnb.predict(mensajeV)
    return "spam" if prediccion[0] == 1 else "No spam"

while(True):
    userMensaje = input("Introduzca su mensaje a detectar: ")
    prediccion = spamDetector(userMensaje)
    print("Este mensaje es:", prediccion)


