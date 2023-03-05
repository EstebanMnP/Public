def data_report(df):
    
    '''
    Dscribe los campos de un DF de pandas de forma clara, dando el tipo de columna, 
    el porcentaje de Missings, la cantidad de valores únicos y el porcentaje de valores únicos respecto 
    al total de valores de esa variable (Cardin)
    Si no hay columnas categóricas, además muestra el valor máximo, el mínimo y el valor medio.
    '''
    # Librerias
    import pandas as pd
    
    # Se saca el nombre de las columnas
    cols = pd.DataFrame(df.columns.values, columns=["COL_N"])

    # Se saca el tipo de los datos
    types = pd.DataFrame(df.dtypes.values, columns=["Data Type"])

    # Se sacan los Missings
    percent_missing = round(df.isnull().sum() * 100 / len(df), 2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns=["Missings (%)"])

    # Se sacan los Valores únicos
    unicos = pd.DataFrame(df.nunique().values, columns=["Unique Values"])
    percent_cardin = round(unicos['Unique Values']*100/len(df), 2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns=["CARDIN (%)"])
    
    try:
        #Se cambia el formato de los datos para que muestren más decimales
        pd.options.display.float_format = '{:.2f}'.format

        # Se sacan los valores mínimos
        min_val_df = pd.DataFrame(df.min().values, columns=["Min. Value"])
        
        # Se sacan los valores máximos
        max_val_df = pd.DataFrame(df.max().values, columns=["Max. Value"])
               
        # Se saca el valor medio
        mean_df = pd.DataFrame(df.mean().values, columns=["Mean Value"])

        # Se concatenan todos los campos
        concatenado = pd.concat([cols, types, percent_missing_df,
                                 unicos, percent_cardin_df, min_val_df, max_val_df, mean_df], axis=1, sort=False)
        concatenado.set_index('COL_N', drop=True, inplace=True)

        return concatenado #el T es un transpon del DF
    
    except:
        
        #Se concatenan todos los campos
        concatenado = pd.concat([cols, types, percent_missing_df, unicos, percent_cardin_df], axis=1, sort=False)
        concatenado.set_index('COL_N', drop=True, inplace=True)

        return concatenado #el T es un transpon del DF
    
def plot_distributions(df):
    """
    Represent the data distribution and check distribution-skewness for each feature
    """
    # Librerías
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Number of plots
    nrows = len(df.columns)

    fig, ax = plt.subplots(nrows, 2, figsize=(15, 40))
    for feature in df:
        idx = df.columns.get_loc(feature)
        
        # distribution
        sns.distplot(df[feature], bins=20, 
                     label='skewness: %.2f'%(df[feature].skew()),
                     ax = ax[idx,0])
        
        # boxplot
        sns.boxplot(df[feature], ax=ax[idx,1])    
    plt.tight_layout()
    plt.show()

def correlation_heatmap(df, limit = 0.3, method='pearson'):

    '''
    Pinta una matriz de correlación triangular inferior que solo muestra los valores por encima de un límite establecido
    El límite por defecto es 0.3
    El método predeterminado para calcular la correlación es el método de pearson
    '''

    # Librerías
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    plt.figure(figsize=(18,12))
    corr = df.corr(method)
    #cmap = sns.diverging_palette(220,10,as_cmap=True)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    sns.heatmap(corr[((corr >= limit) | (corr <= -limit))], annot=True, linewidths=.5, fmt= '.2f', mask=mask)                 
    plt.title('Heatmap displaying the relationship between the features of the data',
              fontsize=20)
    plt.tight_layout()
    plt.show()    

def get_positive_corr_coefficient(df, limit=0.75, method='pearson'):

    ''' 
    Devuelve un DF sólo con las columnas cuya correlación sea menor al límite establecido.
    El límite por defecto es 0.75  y el método de cálculo por defecto es el de pearson
    '''

    # Librerías
    import numpy as np

    positive_corr_df = df.corr(method)
    positive_corr_df = positive_corr_df.mask(np.tril(np.ones(positive_corr_df.shape)).astype(np.bool))
    positive_corr_df = positive_corr_df[abs(positive_corr_df) > limit].stack().reset_index()
    return positive_corr_df

def categorical_columns_proyect_LoL(df):
    ''' 
    Función que limpia las columnas categóricas del proyecto de ML del LoL
    Codifica las columnas de FirstTowerLane y DragnoType para ambos equipos.
    Después de crear las nuevas columnas no categóricas, borra las categóricas
    '''

    # Librerías
    import numpy as np
    import pandas as pd

    #Primero se modifica el nombre de los datos para luego aplicar un get_dummies
    df['blueFirstTowerLane'] = np.where((df.blueFirstTowerLane.str.contains('TOP_LANE'))==True,
                                      "Top_lane",np.where((df.blueFirstTowerLane.str.contains('BOT_LANE'))==True,
                                      "Bot_lane",np.where((df.blueFirstTowerLane.str.contains('MID_LANE'))==True,
                                      "Mid_lane","NaN")))

    df['redFirstTowerLane'] = np.where((df.redFirstTowerLane.str.contains('TOP_LANE'))==True,
                                      "Top_lane",np.where((df.redFirstTowerLane.str.contains('BOT_LANE'))==True,
                                      "Bot_lane",np.where((df.redFirstTowerLane.str.contains('MID_LANE'))==True,
                                      "Mid_lane","NaN")))

    #Se aplica la codificación
    df= pd.get_dummies(df, columns= ["blueFirstTowerLane"], prefix =["blueFirstTowerLane"])
    df= pd.get_dummies(df, columns= ["redFirstTowerLane"], prefix =["redFirstTowerLane"])

    #Se elimina la columna NaN ya que es redundante. 
    #Si no es 1 ninguna de las otras, esta no lo va a ser, será alguna del otro equipo
    df.drop(['blueFirstTowerLane_NaN', "redFirstTowerLane_NaN"], axis=1, inplace =True)

    #Se modifican los datos de la columna con un Where
    df['blueDragonType_Air'] = np.where((df.blueDragnoType.str.contains('AIR_DRAGON'))==True,
                                      1,0)
    df['blueDragonType_Water'] = np.where((df.blueDragnoType.str.contains('WATER_DRAGON'))==True,
                                      1,0)
    df['blueDragonType_Fire'] = np.where((df.blueDragnoType.str.contains('FIRE_DRAGON'))==True,
                                      1,0)
    df['blueDragonType_Earth'] = np.where((df.blueDragnoType.str.contains('EARTH_DRAGON'))==True,
                                      1,0)
    
    #Igual para el equipo rojo
    df['redDragonType_Air'] = np.where((df.redDragnoType.str.contains('AIR_DRAGON'))==True,
                                      1,0)
    df['redDragonType_Water'] = np.where((df.redDragnoType.str.contains('WATER_DRAGON'))==True,
                                      1,0)
    df['redDragonType_Fire'] = np.where((df.redDragnoType.str.contains('FIRE_DRAGON'))==True,
                                      1,0)
    df['redDragonType_Earth'] = np.where((df.redDragnoType.str.contains('EARTH_DRAGON'))==True,
                                      1,0)

    #Se borran las columnas
    df.drop(['blueDragnoType', "redDragnoType"], axis=1, inplace =True)
    
    return df

def inicio_ejecucion():
    # Librería
    from datetime import datetime

    #Inicio de ejecución
    now = datetime.now()
    inicio = now.strftime("%H:%M:%S")
    print("Hora de inicio: ", inicio)

def fin_ejecucion():
    # Librería
    from datetime import datetime

    #Fin de ejecución
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Hora de fin: ", current_time) 

def guardar_modelo(modelo, nombre):
    # Librerías
    import pickle

    # Se guarda el modelo
    pickle.dump(modelo, open(nombre + ".pkl", 'wb'))

def cargar_modelo(nombre):
    # Librerías
    import pickle

    # Se carga el modelo
    modelo = pickle.load(open(nombre + ".pkl", 'rb'))

    return modelo

def dividir_datos(df):
    '''
    Se realizan dos divisiones. Primero una en entrenamiento y validación (80-20)
    Luego vuelve a dividir entrenamiento en train y test 
    '''
    # Librerías
    from sklearn.model_selection import train_test_split

    #Primera división
    datos_entrenamiento, datos_validacion= train_test_split(df, test_size = 0.20, random_state=42)

    #Segunda división
    partidas=datos_entrenamiento.copy()

    X = partidas.copy().drop(['blueWins'], axis=1)
    y = partidas[['blueWins']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Se preparan los datos de validacion tambien
    validacion=datos_validacion.copy()
    X_val = validacion.copy().drop(['blueWins'], axis=1)
    y_val = validacion[['blueWins']]

    return X_train, X_test, y_train, y_test, X_val, y_val