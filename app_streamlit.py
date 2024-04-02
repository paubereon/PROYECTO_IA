
#Importamos la libreria para los modelo
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import ConfusionMatrixDisplay

import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import ipywidgets as widgets
import matplotlib.pyplot as plt

import joblib as jb


#Configuramos la página de Streamlit
import streamlit as st

st.set_page_config(page_title="Predición de deserción de clientes", 
                   page_icon="https://images.freeimages.com/fic/images/icons/61/dragon_soft/512/user.png",
                   layout="centered",
                   initial_sidebar_state="auto")

@st.cache_resource
def load_model():
  modeloNB=jb.load('modeloNB.bin')
  return modeloNB

modeloNB=load_model()

#Definimos el título y la descripción de la aplicación
st.title("Aplicación de predicción")
st.header('Machine Learning para Churn', divider='rainbow')
st.subheader('MODELO CON EL MEJOR SCORE :blue[Naive Bayes]')


with st.container(border=True):
  st.subheader("Modelo Machine Learning para predecir la deserción de clientes")
  #wave es un emojim
  st.write("Realizado por Paula Betina Reyes y Ilona Giovanna Pava :wave:")
  st.write("""
**OBJETIVO**:
El objetivo principal de este proyecto es crear un modelo predictivo que pueda predecir si un cliente decidirá retirarse o continuara.
El DataFrame inicial (df) consta de 14 columnas, siendo la última columna "TARGET CLASS" la que indica si un cliente se retira o continúa. 
Un valor de 1 en esta columna indica que el cliente se retira, mientras que un valor de 0 indica que el cliente continuara.
 """)
  
with st.container( border=True):
  st.subheader("PROYECTO GOOGLE COLAB")
  st.write(""" Enlace de los modelos de Machine Learning entrenados en Google Colab con las librerias de scikit-learn de los tres modelos, las cuales son: Naive Bayes, Árbol de Decisión y Bosques Aleatorios.
        https://colab.research.google.com/drive/1W3ROKwPQ9fVJZh2noxf-K3a19SwMkieD?usp=sharing""")
  


# Cargar los datos
df = pd.read_csv('https://raw.githubusercontent.com/adiacla/bigdata/master/DatosEmpresa.csv')

# Sidebar para aceptar parámetros de entrada
with st.sidebar:
    st.header('Descargar CSV')
    st.write('Haz clic en el siguiente botón para descargar el archivo CSV.')
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button("DatosEmpresaChurn", csv_data, mime='text/csv', file_name='DatosEmpresaChurn.csv')

    

# Preprocesamiento de datos
df['ANTIG'].replace(',', '.', inplace=True)
df['ANTIG'] = df['ANTIG'].astype(float)
df.drop("indice", axis=1, inplace=True)
df.drop(["CATEG", "VISIT"], axis=1, inplace=True)
df.dropna(inplace=True)

# Visualizar valores únicos de la columna 'TARGET CLASS'
st.write("Valores únicos de 'TARGET CLASS':", df["TARGET CLASS"].unique())

# Matriz de correlación
st.subheader('Matriz de correlación')
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(10, 10))
colormap = sns.color_palette("BrBG", 10)
sns.heatmap(corr_matrix, cmap=colormap, annot=True, fmt=".2f", ax=ax)
st.pyplot(fig)

# Eliminar la columna 'ANTIG' debido a su menor varianza
df.drop("ANTIG", axis=1, inplace=True)

# Preparar los datos para el modelo
X = df.drop("TARGET CLASS", axis=1)
y = df["TARGET CLASS"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=454, stratify=df["TARGET CLASS"])

# Entrenar el modelo Naive Bayes
modeloNB = GaussianNB()
modeloNB.fit(X_train, y_train)

# Calcular predicciones y probabilidades
y_pred = modeloNB.predict(X_test)
prob = modeloNB.predict_proba(X_test)

# Score del modelo Naive Bayes
st.subheader('Score del modelo Naive Bayes')
score_naive_bayes = modeloNB.score(X_test, y_test)
st.write("Score del modelo Naive Bayes:", score_naive_bayes)

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, prob[:, 1])  # prob[:, 1] contiene las probabilidades de la clase positiva

# Curva ROC
st.subheader('Curva ROC')
fig_roc, ax_roc = plt.subplots()  # Crear una figura explícita
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("Tasa Falso Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC")
st.pyplot(fig_roc)  # Pasar la figura a st.pyplot()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Calcular la matriz de confusión
matrix = confusion_matrix(y_test, y_pred)

# Matriz de confusión
st.subheader('Matriz de confusión')
fig_confusion, ax_confusion = plt.subplots()  # Crear una figura explícita
display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=modeloNB.classes_)
display.plot(ax=ax_confusion)
st.pyplot(fig_confusion)  # Pasar la figura a st.pyplot()


modeloA=['Naive Bayes']
#Aqui creo un diccionario para indicar cuando el modelo devuleva un número , mostrar el equivalente
# a si se retira o no.

churn = {1 : 'Cliente se retirará', 0 : 'Cliente No se Retirará' }


styleimagen ="<style>[data-testid=stSidebar] [data-testid=stImage]{text-align: center;display: block;margin-left: auto;margin-right: auto;width: 100%;}</style>"
st.sidebar.markdown(styleimagen, unsafe_allow_html=True)

st.sidebar.image("cliente.png", width=300)

#este script es para centrar pero si no lo desea no necesita hacerlo
styletexto = "<style>h2 {text-align: center;}</style>"
st.sidebar.markdown(styletexto, unsafe_allow_html=True)
st.sidebar.header('Seleccione los datos de entrada')

#Vammos a crear una función para mostrar todas las variables laterales para ingresar los datos en el model entrenado
#QAqui vamos a usar varias opciones. Le pasamos por parámetro a la funcion el modelo.

def seleccionar(modeloL):
   #Opción para seleccionar el modelo en un combo box o opción desplegable

  st.sidebar.subheader('Selector de Modelo')
  modeloS=st.sidebar.selectbox("Modelo",modeloL)

  #Filtrar por COMP con un slider
  st.sidebar.subheader('Seleccione la COMP')
  COMPS=st.sidebar.slider("Seleccion",4000,18000,8000,100)
  
  #Filtrar por PROM
  st.sidebar.subheader('Selector del PROM')
  PROMS=st.sidebar.slider("Seleccion",   0.7, 9.0,5.0,.5)
  
  #Filtrar por COMINT
  st.sidebar.subheader('Selector de COMINT')
  COMINTS=st.sidebar.slider("Seleccione",1500,58000,12000,100)
  
  #Filtrar por COMPPRES
  st.sidebar.subheader('Selector de COMPPRES') 
  COMPPRESS=st.sidebar.slider('Seleccione', 17000,90000,25000,100)
  
  #Filtrar por RATE
  st.sidebar.subheader('Selector de RATE')
  RATES=st.sidebar.slider("Seleccione",0.5,4.2,2.0,0.1)

  #Filtrar por DIASSINQ
  st.sidebar.subheader('Selector de DIASSINQ')
  DIASSINQS=st.sidebar.slider("Seleccione", 270,1800,500,10)
  
    #Filtrar por TASARET
  st.sidebar.subheader('Selector de TASARET')
  TASARETS=st.sidebar.slider("Seleccione",0.3,1.9,0.8,.5)
  
    #Filtrar por NUMQ
  st.sidebar.subheader('Selector de NUMQ')
  NUMQS=st.sidebar.slider("Seleccione",3.0,10.0,4.0,0.5)
  
    #Filtrar por departamento
  st.sidebar.subheader('Selector de RETRE entre 3 y 30')
  #RETRES=st.sidebar.slider("Seleccione",3.3,35.0,20.0,.5)
  RETRES=st.sidebar.number_input("Ingrese el valor de RETRE", value=3.3, placeholder="Digite el numero...")
  
  return modeloS,COMPS, PROMS, COMINTS ,COMPPRESS, RATES, DIASSINQS,TASARETS, NUMQS, RETRES
# Se llama la función, y se guardan los valores seleccionados en cada variable

modelo,COMP, PROM, COMINT ,COMPPRES, RATE, DIASSINQ,TASARET, NUMQ, RETRE=seleccionar(modeloA)

#Creamos un container para mostrar los resultados de predicción en el modelo que seleccione

with st.container(border=True):
  st.subheader("Predición")
  st.title("Predicción de Churn")
  st.write(""" El siguiente es el pronóstico de la deserción usando el modelo
           """)
  st.write(modelo)
  st.write("Se han seleccionado los siguientes parámetros:")
  # Presento los parámetros seleccionados en el slidder
  lista=[[COMP, PROM, COMINT ,COMPPRES, RATE, DIASSINQ,TASARET, NUMQ, RETRE]]
  X_predecir=pd.DataFrame(lista,columns=['COMP', 'PROM', 'COMINT', 'COMPPRES', 'RATE', 'DIASSINQ','TASARET', 'NUMQ', 'RETRE'])
  st.dataframe(X_predecir)

modelo=='Naive Bayes'
y_predict=modeloNB.predict(X_predecir)
probabilidad=modeloNB.predict_proba(X_predecir)
importancia=pd.DataFrame()


styleprediccion= '<p style="font-family:sans-serif; color:Green; font-size: 42px;">La predicción es</p>'
st.markdown(styleprediccion, unsafe_allow_html=True)
prediccion='Resultado: '+ str(y_predict[0])+ "    - en conclusion :"+churn[y_predict[0]]
st.header(prediccion+'   :warning:')
  
st.write("Con la siguiente probabilidad")
  
  #Creamos dos columnas para mostrar las probabilidades de la predcción
  # la variable probabilidad es una matriz de dos columnas asi el valor
  # probabilidad[0][0] se refiere a la fila 0, y la columna 0, es decir el primer valor
  # probabilidad[0][1] se refiere a la fila 0, y la columna 1, es decir el segundo valor
 
features=['COMP', 'PROM', 'COMINT', 'COMPPRES', 'RATE', 'DIASSINQ','TASARET', 'NUMQ', 'RETRE']
col1, col2= st.columns(2)
col1.metric(label="Probalidad de NO :", value="{0:.2%}".format(probabilidad[0][0]),delta=" ")
col2.metric(label="Probalidad de SI:", value="{0:.2%}".format(probabilidad[0][1]),delta=" ")
  
st.write("")
if modelo!='Naive Bayes':
    importancia=pd.Series(importancia,index=features)
    st.bar_chart(importancia)  
else:
    st.write("")