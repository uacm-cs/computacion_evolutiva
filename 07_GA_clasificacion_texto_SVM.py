from tqdm import tqdm
import numpy as np
import logging
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn import svm
from sklearn.metrics import f1_score
import unicodedata
import random
import re
import sys
from joblib import Parallel, delayed


# ---------------------------------------------------------------------------------
# Objetivo: Optimización de features para Clasificación Textual
# El siguiente código ejemplifica la optimización de features para
# usarse en como entradas para un clasificador lineal SVM
# Parámetros:
#
# best_Individuo: Se representa como un arreglo binario.
#            1 indica que la feature se incluye en el modelo de entrenamiento
#            0 indica que la feature NO se incluye  en el modelo de entrenamiento
# Vector de features: Pesado TF-IDF
# Función de Aptitud: Puntuación de la medida F1-macro del desempeño del modelo SVM 
#                     en el conjunto de entrenamiento particionado 80/20 (Entrenamiento/Test)  
# Operadores de variación: Cruza y mutación para representación binaria de los best_individuos
# ---------------------------------------------------------------------------------

# Normalización del texto

PUNCTUACTION = ";:,.\\-\"'/"
SYMBOLS = "()[]¿?¡!{}~<>|"
NUMBERS= "0123456789"
SKIP_SYMBOLS = set(PUNCTUACTION + SYMBOLS)
SKIP_SYMBOLS_AND_SPACES = set(PUNCTUACTION + SYMBOLS + '\t\n\r ')



#-----------------------------------------------------
#-----------------------------------------------------
# Funciones para el preprocesamiento del texto
#-----------------------------------------------------
#-----------------------------------------------------


_STOPWORDS = stopwords.words("spanish")  # agregar más palabras a esta lista si es necesario
LEncoder = LabelEncoder()

#-------------------------------
def normaliza_texto(input_str,
                    punct=False,
                    accents=False,
                    num=False,
                    max_dup=2):
    """
        punct=False (elimina la puntuación, True deja intacta la puntuación)
        accents=False (elimina los acentos, True deja intactos los acentos)
        num= False (elimina los números, True deja intactos los acentos)
        max_dup=2 (número máximo de símbolos duplicados de forma consecutiva, rrrrr => rr)
    """
    
    nfkd_f = unicodedata.normalize('NFKD', input_str)
    n_str = []
    c_prev = ''
    cc_prev = 0
    for c in nfkd_f:
        if not num:
            if c in NUMBERS:
                continue
        if not punct:
            if c in SKIP_SYMBOLS:
                continue
        if not accents and unicodedata.combining(c):
            continue
        if c_prev == c:
            cc_prev += 1
            if cc_prev >= max_dup:
                continue
        else:
            cc_prev = 0
        n_str.append(c)
        c_prev = c
    texto = unicodedata.normalize('NFKD', "".join(n_str))
    texto = re.sub(r'(\s)+', r' ', texto.strip(), flags=re.IGNORECASE)
    return texto

#-------------------------------
# Preprocesamiento personalizado 
def mi_preprocesamiento(texto):
    #convierte a minúsculas el texto antes de normalizar
    tokens = word_tokenize(texto.lower())
    texto = " ".join(tokens)
    texto = normaliza_texto(texto)
    return texto

#-------------------------------    
# Tokenizador personalizado 
def mi_tokenizador(texto):
    # Elimina stopwords: palabras que no se consideran de contenido Y que no agregan valor semántico al texto
    texto = [t for t in texto.split() if t not in _STOPWORDS]    
    return texto

#------------------------------------------------------------------------
def cargar_datos(train_file, test_file):
    """Carga Y prepara los datos de texto para el experimento"""
    print(f"Cargando datos...")
    dataset_train = pd.read_json(train_file, lines=True)
    dataset_test = pd.read_json(test_file, lines=True)

    X_train = dataset_train['text'].to_numpy()
    Y_train = dataset_train['klass'].to_numpy()

    X_test = dataset_test['text'].to_numpy()
    Y_test = dataset_test['klass'].to_numpy()

    Y_train= LEncoder.fit_transform(Y_train)
    Y_test= LEncoder.transform(Y_test)
    
    return X_train, X_test, Y_train, Y_test

#------------------------------------------------------------------------
def vectorizar_texto(train_data, test_data):
    print(f"Vectorizando datos...")
    N_GRAMS = 1
    vectorizador = TfidfVectorizer(analyzer="word", preprocessor=mi_preprocesamiento, tokenizer=mi_tokenizador,  ngram_range=(1,N_GRAMS))
    X_train = vectorizador.fit_transform(train_data)
    total_features = len(vectorizador.get_feature_names_out())

    # Reduce al 20% el total de componentes del vector basadas en su frecuencia
    features_reducidas = int(total_features * 0.20)
    # Se vuelve a configurar el vectorizador con la features reducidas: parámetro max_features 
    vectorizador = TfidfVectorizer(analyzer="word", preprocessor=mi_preprocesamiento, tokenizer=mi_tokenizador,  ngram_range=(1,N_GRAMS), max_features=features_reducidas)
    X_train = vectorizador.fit_transform(train_data)

    # Se tranforma el conjunto de prueba al mismo espacio de representación del conjunto de entrenamiento
    X_test = vectorizador.transform(test_data)

    return X_train, X_test, vectorizador.get_feature_names_out()


#------------------------------------------------------------------------
#------------------------------------------------------------------------
# Funciones del algoritmo genético
#------------------------------------------------------------------------
#------------------------------------------------------------------------
def inicializar_poblacion(tam_poblacion, num_features):
    """Inicializa una población de best_individuos binarios aleatorios"""
    print(f"Inicializando población...")
    # Genera los vectores de cadenas binarias que representan la selección de la features (1) o no la seleccionan (0)
    return np.random.randint(0, 2, size=(tam_poblacion, num_features))

#-------------------------------
def funcion_fitness(best_individuo, X, Y, kfolds=0):
    """Evalúa la aptitud de un best_individuo usando validación cruzada o partición kfolds=0"""

    # Obtener los índices de las características seleccionadas    
    # Retorna los índices de las features que se seleccionaron
    # np.where devuelve una tupla
    features_seleccionados = np.where(best_individuo == 1)[0]

    if len(features_seleccionados) == 0:
        return 0.0
    
    # Solo selecciona del conjunto de datos a las características seleccionadas

    X_reducida = X[:, features_seleccionados]
    
    clf = svm.SVC(kernel='linear')

    # Calcular la precisión con validación cruzada
    if kfolds == 0:
        # División de datos: 80/20
        X_train, X_test, Y_train, Y_test =  train_test_split(X_reducida, Y, test_size=0.2, stratify= Y, random_state=42)
        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)
        score = f1_score(Y_test, y_pred, average="macro")
    else:
        # División de datos: 3 folds
        scores = cross_val_score(clf, X_reducida, Y, cv=kfolds, scoring='f1_macro')
        score = np.mean(scores)
    return score

#------------------------------------------------------------------------
def evaluar_poblacion(poblacion, X, Y):
    """Evalúa todos los best_individuos de la población"""
    print(f'Evaluando población ...')
    fitness = []
    for k, ind in tqdm(zip(range(len(poblacion)), poblacion), total=len(poblacion)):
        # print(f'Evaluando best_individuo {k}')
        score = funcion_fitness(ind, X, Y)
        fitness.append(score)
    return  np.array(fitness)

#------------------------------------------------------------------------
def evaluar_best_individuo(ind, X, Y):
    return  funcion_fitness(ind, X, Y)

#------------------------------------------------------------------------
def evaluar_poblacion_paralelo(poblacion, X, Y, n_jobs=-1):
    """Evalúa la población en paralelo usando multiprocessing."""
    print(f"Evaluando población en paralelo ({n_jobs} núcleos)...")
    # Configurar el número de workers (n_jobs=-1 usa todos los núcleos)
    n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    # Evaluar en paralelo

    resultados = Parallel(
    n_jobs = n_jobs-1,
    timeout = 300,  # 5min
    verbose = 10
    )(
        delayed(funcion_fitness)(ind, X, Y)
        for ind in poblacion
    )

    fitness = np.array(resultados)
    
    return fitness


#------------------------------------------------------------------------
def seleccionar_padres(poblacion, aptitudes):
    """Selecciona dos padres mediante torneo."""
    torneo = random.sample(list(zip(poblacion, aptitudes)), k=4)
    torneo.sort(key=lambda x: x[1])  
    return torneo[0][0], torneo[1][0]


#------------------------------------------------------------------------
def elitismo(poblacion, fitness, tam_elite=2):
    """Selecciona los 'tam_elite' mejores best_individuos."""
    # Ordenar por fitness (mayor = mejor)
    ranked_indices = np.argsort(fitness)[::-1]
    elites = [poblacion[i] for i in ranked_indices[:tam_elite]]
    return elites

#------------------------------------------------------------------------
def cruzar(padre1, padre2):
    #Cruza uniforme
    probabilidad_intercambio=0.5
    # Generar máscara de intercambio
    mascara = np.random.random(len(padre1)) < probabilidad_intercambio
    # Crear hijos
    hijo1 = np.where(mascara, padre2, padre1)
    hijo2 = np.where(mascara, padre1, padre2)
    return hijo1, hijo2

#------------------------------------------------------------------------
def mutar(best_individuo, tasa_mutacion=0.1):
    # Crea el indice de las mutaciones que se deben aplicar
    indices_mutaciones = np.random.rand(len(best_individuo)) < tasa_mutacion
    # Muta el gen: si es 1 - [1] = 0 ; 1 - [0] = 1;  [VALOR_best_INDIVIDUO]
    best_individuo[indices_mutaciones] = 1 - best_individuo[indices_mutaciones]
    return best_individuo

#------------------------------------------------------------------------
def algoritmo_genetico(X, Y, tam_poblacion=50, num_generaciones=20, run_paralelo=False):
    """Algoritmo genético principal para selección de características"""
    logging.debug(f"Iniciando GA  ...")
    num_features = X.shape[1]
    poblacion = inicializar_poblacion(tam_poblacion, num_features)
    best_fitness_hist = []
    mean_fitness_hist = []

    for generacion in  tqdm(range(num_generaciones)):
        # Evaluar la población actual
        if run_paralelo:
            val_fitness = evaluar_poblacion_paralelo(poblacion, X, Y)
        else:    
            val_fitness = evaluar_poblacion(poblacion, X, Y)
        
        # Registrar estadísticas
        print("obteniendo mejor fitness" )
        best_fitness = np.max(val_fitness)
        print("obteniendo media fitness " )
        mean_fitness = np.mean(val_fitness)
        best_fitness_hist.append(best_fitness)
        mean_fitness_hist.append(mean_fitness)
        
        print(f"Generación {generacion + 1}: Mejor fitness = {best_fitness:.4f}, Fitness promedio = {mean_fitness:.4f}")

        nueva_poblacion = []
        fitness_nueva_poblacion = []
        for _ in range(tam_poblacion // 2):
            # Seleccionar padres
            padre1, padre2 = seleccionar_padres(poblacion, val_fitness)            
            # Crear descendencia mediante cruce
            hijo1, hijo2 = cruzar(padre1, padre2)
            hijo1 = mutar(hijo1)
            hijo2 = mutar(hijo2)
            nueva_poblacion.append(hijo1)            
            nueva_poblacion.append(hijo2)        
        
        
        #-----------------
        # Población: Los hijos sustituyen a los padres
        #-----------------
        poblacion = np.array(nueva_poblacion)

        #-----------------
        # Población con elitismo de padres y parte de los hijos
        #-----------------
        # nueva_poblacion = np.array(nueva_poblacion)
        # K_best_padres = 5
        # poblacion[:K_best_padres, ] = elitismo(poblacion, val_fitness, K_best_padres)
        # poblacion[K_best_padres:, ] = nueva_poblacion[K_best_padres:, ]

    
    # Seleccionar el mejor best_individuo final
    if run_paralelo:
            val_fitness = evaluar_poblacion_paralelo(poblacion, X, Y)
    else:    
            val_fitness = evaluar_poblacion(poblacion, X, Y)
    max_fitness = np.argmax(val_fitness)
    best_individuo = poblacion[max_fitness]

    return best_individuo, best_fitness_hist, mean_fitness_hist


#--------------------------------------------------------------------
# Función para predecir datos con el mejor best_individuo obtenido
# por el algoritmo genético
#--------------------------------------------------------------------
def predecir_datos(best_individuo,  X, Y, X_test):
    # Obtener los índices de las características seleccionadas
    print(f"Prediciendo datos con el conjunto de TEST ...")
    features_seleccionados = np.where(best_individuo == 1)[0]
    print(f"features seleccionadas: {features_seleccionados.shape}")
    print(f"X: {X.shape}")

    if len(features_seleccionados) == 0:
        return 0.0
    
    # Reducir el conjunto de datos a las características seleccionadas
    X_reducida = X[:, features_seleccionados]
    print(f"X_reducida: {X_reducida.shape}")

    # Seleccionar el clasificador
    clf = svm.SVC(kernel='linear')
    # Predecir datos con el modelo creado (Features por clasificar)
    print("Entrenando con el mejor best_individuo de la evolución")
    clf.fit(X_reducida, Y)

    print("Prediciendo ...")
    X_test_reduced = X_test[:, features_seleccionados]
    y_pred = clf.predict(X_test_reduced)
    
    return y_pred 


#------------------------------------------------------------------------
# Programa principal
#------------------------------------------------------------------------
if __name__ == '__main__':
    
    # Cargar y vectorizar los datos
    tam_poblacion=50
    num_generaciones=5
    paralelo = True
    X_train_data, X_test_data, Y_train_data, Y_test_data = cargar_datos(train_file="./data/dataset_polaridad_es_train.json", 
                                                                        test_file="./data/dataset_polaridad_es_test.json")
    X_train, X_test, nombre_features = vectorizar_texto(X_train_data, X_test_data)
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    Y_train, Y_test =  Y_train_data,  Y_test_data

    best_individuo, best_hist, mean_hist = algoritmo_genetico(X_train, Y_train,
                                                                tam_poblacion=tam_poblacion, num_generaciones=num_generaciones, run_paralelo=paralelo)
    # Mostrar características seleccionadas
    features_seleccionados = np.where(best_individuo == 1)[0]
    print(f"\nCaracterísticas seleccionadas para SVM: {len(features_seleccionados)}/{X_train.shape[1]}")

    y_pred  = predecir_datos(best_individuo, X_train, Y_train, X_test)
    score = f1_score(Y_test, y_pred, average="macro")
    print(f"F1-macro en el conjunto de Test: {score}")
