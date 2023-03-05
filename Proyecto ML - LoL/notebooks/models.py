class Models:

    # Atributos comunes
    # random_state = 42

    def __init__(self, X_train, X_test, y_train, y_test) -> None:
        #Datos de entrenamiento
        self.X_train = X_train
        self.X_test = X_test

        #Datos de test
        self.y_train = y_train
        self.y_test = y_test

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

    def predecir(self, modelo):
        #Librerías
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        #Se hace la predicción para train
        pred_train = modelo.predict(self.X_train)

        #Se hace la predicción para test
        pred_test = modelo.predict(self.X_test)

        #Se imprimen las métricas de error sobre train
        Accuracy_train = accuracy_score(self.y_train, pred_train)
        print("-------------------------")
        print('Accuracy train', Accuracy_train)

        #Se imprimen las métricas de error sobre test
        Accuracy_test = accuracy_score(self.y_test, pred_test)
        print('Accuracy test', Accuracy_test)
        print("-------------------------")
        
        # Se calculan otras métricas
        c_matrix = confusion_matrix(self.y_test, pred_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix)
        report = classification_report(self.y_test, pred_test)

        #Se muestran las otras métricas
        print("-------------------------")
        print("Matriz de confusión")
        disp.plot(cmap = "winter")
        plt.show()
        print("-------------------------")
        
        print("-------------------------")
        print("Report")
        print(report)
        print("-------------------------")

        return pred_train, pred_test, Accuracy_train, Accuracy_test
    
    def predecir_sin_print(self, modelo):
        #Librerías
        from sklearn.metrics import accuracy_score

        #Se hace la predicción para train
        pred_train = modelo.predict(self.X_train)

        #Se hace la predicción para test
        pred_test = modelo.predict(self.X_test)

        #Se calculan las métricas de error sobre train
        Accuracy_train = accuracy_score(self.y_train, pred_train)
        
        #Se calculan las métricas de error sobre test
        Accuracy_test = accuracy_score(self.y_test, pred_test)
        
        return pred_train, pred_test, Accuracy_train, Accuracy_test

    def predecir_validacion(modelo, X_validacion, y_validacion):
        #Librerías
        from sklearn.metrics import accuracy_score

        #Se hace la predicción
        predicciones = modelo.predict(X_validacion)

        #Se imprimen las métricas de error sobre test
        Accuracy = accuracy_score(y_validacion, predicciones)
        print('Accuracy validación', Accuracy)

        return Accuracy
            
    def guardar_modelo(self, modelo, nombre):
        # Librerías
        import pickle

        # Se guarda el modelo
        pickle.dump(modelo, open(nombre, 'wb'))
    
    def cargar_modelo(self, nombre):
        # Librerías
        import pickle

        # Se carga el modelo
        modelo = pickle.load(open(nombre, 'rb'))

        return modelo

    def matriz_recopilar():
        # Librerias
        import pandas as pd

        #Se crea un DF con los resultados de Accuracy
        dfAccuracy = pd.DataFrame(columns = ["Modelo","Train_15min", "Test_15min","Validación_15min", "Train_10min", "Test_10min", "Validación_10min"])  

        return dfAccuracy
        
    def recopilar_resultados(dfAccuracy, modelo, Accuracy_train_15min, Accuracy_test_15min, Accuracy_train_10min, Accuracy_test_10min, Accuracy_val_15min,Accuracy_val_10min ):
        # Librerias
        import pandas as pd

        #Se añaden los nuevos datos
        nueva_fila = pd.Series([modelo, Accuracy_train_15min ,Accuracy_test_15min,Accuracy_val_15min,
                                Accuracy_train_10min,Accuracy_test_10min,Accuracy_val_10min], 
                                index = ["Modelo","Train_15min", "Test_15min","Validación_15min", "Train_10min", "Test_10min", "Validación_10min"])
        dfAccuracy = dfAccuracy.append(nueva_fila, ignore_index=True )

        return dfAccuracy

    def lazyclassifier(self):
        # Librerias
        from lazypredict.Supervised import LazyClassifier

        #LazyClassifier
        clf = LazyClassifier(random_state = 42)
        models,predictions = clf.fit(self.X_train, self.X_test, self.y_train, self.y_test)

        return models

    def randomforest(self, n_estimators = 200, random_state = 42, metricas = True):

        # Librerias
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        # Inicio de la ejecución
        Models.inicio_ejecucion()

        #Se genera el modelo
        ranforest = RandomForestClassifier(n_estimators = n_estimators, random_state = random_state)

        #Se modifica la forma() de y para el modelo
        y_train= np.ravel(self.y_train)

        #Se entrena el modelo
        modelo_entrenado = ranforest.fit(self.X_train, y_train)

        #Fin de ejecución
        Models.fin_ejecucion()

        if metricas == True:

            #Se utiliza el modelo para predecir
            pred_train, pred_test, Accuracy_train, Accuracy_test = self.predecir(modelo_entrenado)

            return modelo_entrenado, pred_train, pred_test, Accuracy_train, Accuracy_test
        
        elif metricas == False:
            
            #Se utiliza el modelo para predecir
            pred_train, pred_test, Accuracy_train, Accuracy_test = self.predecir_sin_print(modelo_entrenado)

            return modelo_entrenado, pred_train, pred_test, Accuracy_train, Accuracy_test
    
    def randomforest_GridS(self, random_state = 42):

        # Librerias
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
        import numpy as np
        
        # Inicio de la ejecución
        Models.inicio_ejecucion()

        #Se genera el modelo
        ranforest = RandomForestClassifier(random_state = random_state)

        #Se buscan los mejores hyperparámetros
        params = {'criterion': ["gini", "entropy", "log_loss"] , 
                        "n_estimators": [100,150,200],
                        "max_depth":np.arange(2,8,1),
                        "min_samples_split":np.arange(5,8,1), 
                        "min_samples_leaf":np.arange(5,8,1),
                        "max_features": ["sqrt", 3, 4,6]
                        }

        ranforest_opt = GridSearchCV(ranforest, 
                                    param_grid=params,
                                    cv=5, #Folds (grupos) del cross validation
                                    n_jobs=-1)
        
        #Se modifica la forma() de y para el modelo
        y_train= np.ravel(self.y_train)

        #Se entrena el modelo
        modelo_entrenado = ranforest_opt.fit(self.X_train, y_train)

        #Fin de ejecución
        Models.fin_ejecucion()

        #Se utiliza el modelo para predecir
        pred_train, pred_test, Accuracy_train, Accuracy_test = self.predecir(modelo_entrenado)

        return modelo_entrenado, pred_train, pred_test, Accuracy_train, Accuracy_test

    def gradientboosting(self, n_estimators = 200, random_state = 42):

        # Librerias
        from sklearn.ensemble import GradientBoostingClassifier
        
        # Inicio de la ejecución
        Models.inicio_ejecucion()

        #Se genera el modelo
        gradientboost = GradientBoostingClassifier(n_estimators = n_estimators, random_state = random_state)

        #Se entrena el modelo
        modelo_entrenado = gradientboost.fit(self.X_train, self.y_train)

        #Fin de ejecución
        Models.fin_ejecucion()

        #Se utiliza el modelo para predecir
        pred_train, pred_test, Accuracy_train, Accuracy_test = self.predecir(modelo_entrenado)

        return modelo_entrenado, pred_train, pred_test, Accuracy_train, Accuracy_test

    def gradientboosting_GridS(self, random_state = 42):

        # Librerias
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import GridSearchCV
        import numpy as np
        
        # Inicio de la ejecución
        Models.inicio_ejecucion()

        #Se genera el modelo
        gradientboost = GradientBoostingClassifier(random_state = random_state)

        #Se buscan los mejores hyperparámetros
        params = {"n_estimators": [100,150,200],
                        "max_depth":np.arange(2,8,1),
                        "min_samples_split":np.arange(5,8,1), 
                        "min_samples_leaf":np.arange(5,8,1),
                        "max_features": ["sqrt", 3, 4,6]
                        }

        gradientboost_opt = GridSearchCV(gradientboost, 
                                    param_grid=params,
                                    cv=5, #Folds (grupos) del cross validation
                                    n_jobs=-1)

        #Se entrena el modelo
        modelo_entrenado = gradientboost_opt.fit(self.X_train, self.y_train)

        #Fin de ejecución
        Models.fin_ejecucion()

        #Se utiliza el modelo para predecir
        pred_train, pred_test, Accuracy_train, Accuracy_test = self.predecir(modelo_entrenado)

        return modelo_entrenado, pred_train, pred_test, Accuracy_train, Accuracy_test

    def catboost(self, n_estimators = 200, random_state = 42):

        # Librerias
        from catboost import CatBoostClassifier, Pool
        
        # Inicio de la ejecución
        Models.inicio_ejecucion()

        #Se genera el modelo
        castboost = CatBoostClassifier(n_estimators = n_estimators, random_state = random_state)

        #Se crea el Pool
        pool_train=Pool(self.X_train, self.y_train)

        #Se entrena el modelo
        modelo_entrenado = castboost.fit(pool_train)

        #Fin de ejecución
        Models.fin_ejecucion()

        #Se utiliza el modelo para predecir
        pred_train, pred_test, Accuracy_train, Accuracy_test = self.predecir(modelo_entrenado)

        return modelo_entrenado, pred_train, pred_test, Accuracy_train, Accuracy_test
    
    def catboost_GridS(self):

        # Librerias
        from catboost import CatBoostClassifier, Pool
        import numpy as np        
        
        # Inicio de la ejecución
        Models.inicio_ejecucion()

        #Se genera el modelo
        modelo_entrenado = CatBoostClassifier()

        #Se crea el Pool
        pool_train=Pool(self.X_train, self.y_train)

        #Se definen los hyperparámetros
        params = {"random_state":[42],
                    "n_estimators":[100,150,200],
                    "depth":[3,5,7],
                    "learning_rate":[0.05,0.1,0.15,0.2,0.5], 
                    'l2_leaf_reg': np.logspace(-20, -19, 3),
                    'leaf_estimation_iterations': [10]
                    }

        #Se sacan los mejores parámetros y el modelo sale ya entrenado
        dict_modelo_entrenado  = modelo_entrenado.grid_search(param_grid=params,
                                            X=pool_train,
                                            cv=5)
        
        #Fin de ejecución
        Models.fin_ejecucion()

        #Se utiliza el modelo para predecir
        pred_train, pred_test, Accuracy_train, Accuracy_test = self.predecir(modelo_entrenado)

        return modelo_entrenado, pred_train, pred_test, Accuracy_train, Accuracy_test
    
    def xgboost(self, n_estimators = 200, random_state = 42):

        # Librerias
        from xgboost import XGBClassifier     
        
        # Inicio de la ejecución
        Models.inicio_ejecucion()

        #Se genera el modelo
        xgboost = XGBClassifier(n_estimators = n_estimators, random_state = random_state)

        #Se entrena el modelo
        modelo_entrenado = xgboost.fit(self.X_train, self.y_train)

        #Fin de ejecución
        Models.fin_ejecucion()

        #Se utiliza el modelo para predecir
        pred_train, pred_test, Accuracy_train, Accuracy_test = self.predecir(modelo_entrenado)

        return modelo_entrenado, pred_train, pred_test, Accuracy_train, Accuracy_test
    
    def lightgbm(self, n_estimators = 200, random_state = 42):

        # Librerias
        from lightgbm import LGBMClassifier    
        
        # Inicio de la ejecución
        Models.inicio_ejecucion()

        #Se genera el modelo
        lightgbm = LGBMClassifier(n_estimators = n_estimators, random_state = random_state)

        #Se entrena el modelo
        modelo_entrenado = lightgbm.fit(self.X_train, self.y_train)

        #Fin de ejecución
        Models.fin_ejecucion()

        #Se utiliza el modelo para predecir
        pred_train, pred_test, Accuracy_train, Accuracy_test = self.predecir(modelo_entrenado)

        return modelo_entrenado, pred_train, pred_test, Accuracy_train, Accuracy_test
    
    def lightgbm_GridS(self, random_state = 42):

        # Librerias
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import GridSearchCV
        import numpy as np        
        
        # Inicio de la ejecución
        Models.inicio_ejecucion()

        #Se genera el modelo
        lightgbm = LGBMClassifier(random_state = random_state)

        #Se buscan los mejores hyperparámetros
        params= {'min_child_weight': [0.001,0.01,0.1, 1, 5, 10],
                        "n_estimators": [100,150,200],
                        "max_depth":np.arange(3,8,1),
                        "learning_rate":[0.05,0.1,0.2,0.5,0.7,0.8,1],
                        }

        lightgbm_opt = GridSearchCV(lightgbm, 
                                    param_grid=params,
                                    cv=5, #Folds (grupos) del cross validation
                                    n_jobs=-1)
        
        #Se entrena el modelo
        modelo_entrenado = lightgbm_opt.fit(self.X_train, self.y_train)

        #Fin de ejecución
        Models.fin_ejecucion()

        #Se utiliza el modelo para predecir
        pred_train, pred_test, Accuracy_train, Accuracy_test = self.predecir(modelo_entrenado)

        return modelo_entrenado, pred_train, pred_test, Accuracy_train, Accuracy_test
    
    def svc(self,random_state = 42):

        # Librerias
        from sklearn.svm import SVC      
        
        # Inicio de la ejecución
        Models.inicio_ejecucion()

        #Se genera el modelo
        svc = SVC(random_state = random_state)

        #Se entrena el modelo
        modelo_entrenado = svc.fit(self.X_train, self.y_train)

        #Fin de ejecución
        Models.fin_ejecucion()

        #Se utiliza el modelo para predecir
        pred_train, pred_test, Accuracy_train, Accuracy_test = self.predecir(modelo_entrenado)

        return modelo_entrenado, pred_train, pred_test, Accuracy_train, Accuracy_test

    def svc_GridS(self,random_state = 42):

        # Librerias
        from sklearn.svm import SVC, LinearSVC
        from sklearn.model_selection import GridSearchCV     
        
        # Inicio de la ejecución
        Models.inicio_ejecucion()

        #Se genera el modelo
        svc = SVC(random_state = random_state)

        #Se buscan los mejores hyperparámetros
        params = {'C': [0.1, 1, 10, 100, 1000], 
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001, "auto"]
                        }

        SVC_opt = GridSearchCV(svc, 
                                    param_grid=params,
                                    cv=5, #Folds (grupos) del cross validation
                                    n_jobs=-1)

        #Se entrena el modelo
        modelo_entrenado = SVC_opt.fit(self.X_train, self.y_train)

        #Fin de ejecución
        Models.fin_ejecucion()

        #Se utiliza el modelo para predecir
        pred_train, pred_test, Accuracy_train, Accuracy_test = self.predecir(modelo_entrenado)

        return modelo_entrenado, pred_train, pred_test, Accuracy_train, Accuracy_test
    
    def linearsvc(self,random_state = 42):

        # Librerias
        from sklearn.svm import LinearSVC      
        
        # Inicio de la ejecución
        Models.inicio_ejecucion()

        #Se genera el modelo
        linearsvc = LinearSVC(random_state = random_state)

        #Se entrena el modelo
        modelo_entrenado = linearsvc.fit(self.X_train, self.y_train)

        #Fin de ejecución
        Models.fin_ejecucion()

        #Se utiliza el modelo para predecir
        pred_train, pred_test, Accuracy_train, Accuracy_test = self.predecir(modelo_entrenado)

        return modelo_entrenado, pred_train, pred_test, Accuracy_train, Accuracy_test
    
    def linearsvc_GridS(self,random_state = 42):

        # Librerias
        from sklearn.svm import LinearSVC      
        from sklearn.model_selection import GridSearchCV 

        # Inicio de la ejecución
        Models.inicio_ejecucion()

        #Se genera el modelo
        linearsvc = LinearSVC(random_state = random_state) 

        #Se buscan los mejores hyperparámetros
        params = {'C': [0.1, 1, 10, 100, 1000], 
                            'max_iter': [1000,5000,10000]
                        }

        linearsvc_opt = GridSearchCV(linearsvc, 
                                    param_grid=params,
                                    cv=5, #Folds (grupos) del cross validation
                                    n_jobs=-1)   
        
        #Se entrena el modelo
        modelo_entrenado = linearsvc_opt.fit(self.X_train, self.y_train)

        #Fin de ejecución
        Models.fin_ejecucion()

        #Se utiliza el modelo para predecir
        pred_train, pred_test, Accuracy_train, Accuracy_test = self.predecir(modelo_entrenado)

        return modelo_entrenado, pred_train, pred_test, Accuracy_train, Accuracy_test
    
    
