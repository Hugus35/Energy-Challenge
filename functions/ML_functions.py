from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV, KFold, learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import xgboost as xgb

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd


def weighted_accuracy(y_true, y_pred):
    """Compute the weighted accuracy imposed by the rules of the challenge

    Args:
        y_true (list): True target values
        y_pred (list): Target values predicted by the model

    Returns:
        float: Weighted accuracy from both lists
    """    
    return accuracy_score(np.sign(y_true), np.sign(y_pred), sample_weight = np.abs(y_true))


def optimization(X, y, model, params, my_score, verbose = 3):
    """ Use a GridSearchCV to optimize the model with a cross-validation on X and y

    Args:
        X (pd.DataFrame, list, np.arrays): Dataset containing features
        y (pd.DataFrame, list, np.arrays): Target variable corresponding to X
        model (estimator object): Model to optimize
        params (dict): Dictionnary containing values of hyperparameters to try. 
                        Keys are name of the hyperparameters, values are hyperparameter values to try
        my_score (callable): Score to use for the cross_validation 
        verbose (int, optional): Details to print during the optimization. Defaults to 3.

    Returns:
        best_params (dict): Best parameters combinations found.
        best_model(estimator object) : Model with best parameters settings.
    """    
    grid = GridSearchCV(model, params, cv = 5, scoring = my_score, verbose = verbose, return_train_score = True)
    grid.fit(X, y)
    
    best_params = grid.best_params_
    best_model = grid.best_estimator_
    
    return best_params, best_model



def fit_models(X, y, verbose, score, shuffle = False, models = ['forest', 'KNN', 'SVM', 'XGB', 'regression']):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = shuffle, random_state = 1)
    trained_models = []

    # RANDOM FOREST
    if 'forest' in models :
        forest = RandomForestRegressor()

        params = {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 20],
            'min_samples_split': [10],
            'min_samples_leaf': [1, 4]
            }

        grid = GridSearchCV(forest, params, cv = 5, scoring = score, verbose = verbose, return_train_score = True)
        grid.fit(X_train, y_train)
        
        best_params = grid.best_params_
        
        forest = RandomForestRegressor()
        forest.set_params(**best_params)
        forest.fit(X_train, y_train)
        print("Random Forest :\n")
        print(f"Best params : {best_params}\n")
        print_scores(X_train, y_train, X_test, y_test, forest)
        trained_models.append(forest)


    # KNN
    if 'KNN' in models:
        KNN = KNeighborsRegressor()
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        params = {
            'n_neighbors': np.arange(1, 10, 1),
            }

        grid = GridSearchCV(KNN, params, cv = 5, scoring = score, verbose = verbose, return_train_score = True)
        grid.fit(X_train_scaled, y_train)
        
        best_params = grid.best_params_
        
        KNN = KNeighborsRegressor()
        KNN.set_params(**best_params)
        KNN.fit(X_train_scaled, y_train)
        print("KNN :\n")
        print(f"Best params : {best_params}\n")
        print_scores(X_train_scaled, y_train, X_test_scaled, y_test, KNN)
        trained_models.append(KNN)
    

    # SVM
    if 'SVM' in models:
        SVM = SVR()
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        params = {
            'C': [1, 10, 100],         # Valeurs de régularisation
            'epsilon': [0.01, 0.1],    # Marge
            'kernel': ['poly', 'linear'],    # Noyau
            #'gamma': ['auto'],      # Coefficients du noyau RBF
            'degree' : [2, 3]
        }

        grid = GridSearchCV(SVM, params, cv = 5, scoring = score, verbose = verbose, return_train_score = True)
        grid.fit(X_train_scaled, y_train)
        
        best_params = grid.best_params_
        
        SVM = SVR()
        SVM.set_params(**best_params)
        SVM.fit(X_train_scaled, y_train)
        print("SVR :\n")
        print(f"Best params : {best_params}\n")
        print_scores(X_train_scaled, y_train, X_test_scaled, y_test, SVM)
        trained_models.append(SVM)


    # XGB
    if 'XGB' in models:
        XGB = xgb.XGBRegressor()

        params = {
            "n_estimators": [10, 50],        # Nombre d'arbres
            "learning_rate": [0.01, 0.1, 0.2],     # Taux d'apprentissage
            "max_depth": [3, 5],                # Profondeur maximale des arbres
            "subsample": [0.8, 1.0],               # Fraction des échantillons utilisés par arbre
            "colsample_bytree": [0.8, 1.0],        # Fraction des colonnes utilisées par arbre
            "reg_alpha": [0, 0.1, 1],              # Régularisation L1
            }

        grid = GridSearchCV(XGB, params, cv = 5, scoring = score, verbose = verbose, return_train_score = True)
        grid.fit(X_train_scaled, y_train)
        
        best_params = grid.best_params_
        
        XGB = xgb.XGBRegressor()
        XGB.set_params(**best_params)
        XGB.fit(X_train_scaled, y_train)
        print("XGB :\n")
        print(f"Best params : {best_params}\n")
        print_scores(X_train, y_train, X_test, y_test, XGB)
        trained_models.append(XGB)

    # Linear Regression
    if 'regression' in models:
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        print("Linear Regression :\n")
        print_scores(X_train, y_train, X_test, y_test, reg)
        trained_models.append(reg)


    return trained_models



def plot_correlation(y_train):
    plt.figure()
    plt.ylabel("Load forecast")
    y_train.plot()
    plot_acf(y_train)
    plot_pacf(y_train)



def plot_learning(model, X, y, my_score):

    train_sizes, train_scores, test_scores = learning_curve(
        estimator = model,
        X=X,
        y=y,
        cv=3,  # Nombre de plis pour la validation croisée
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring = my_score,
        shuffle = False
    )
    for size, train, test in zip(train_sizes, np.mean(train_scores, axis=1), np.mean(test_scores, axis=1)):
        print(f"Taille : {size:.0f} | train_score : {train:.4f} | test_score : {test:.4f}")

    # Calculer les moyennes et les écarts types
    train_scores_mean = np.mean(train_scores, axis=1)
    #train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    #test_scores_std = np.std(test_scores, axis=1)

    # Tracer la courbe d'apprentissage
    plt.figure()
    #plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    #plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score d'entraînement")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score de validation")
    plt.title('Courbe d\'apprentissage')
    plt.xlabel('Fraction de l\'ensemble d\'entraînement')
    plt.ylabel('Score')
    plt.legend(loc="best")
    plt.show()


def print_scores(X_train, y_train, X_test, y_test, model):
    """Print different scores from the model predictions

    Args:
        X_train (pd.DataFrame or np.array): Dataset used to train the model
        y_train (np.array or list): Labels corresponding to X_train
        X_test (pd.DataFrame or np.array): Dataset to test the model
        y_test (np.array or list): Labels corresponding to X_test
        model (estimator object): Model trained on X_train, y_train to evaluate
    """    
    
    print(f"Weighted_accuracy | Train : {weighted_accuracy(y_train, model.predict(X_train))}")
    print(f"Weighted_accuracy | Test : {weighted_accuracy(y_test, model.predict(X_test))}")

    print(f"Accuracy | Train : {accuracy_score(np.sign(y_train), np.sign(model.predict(X_train)))}")
    print(f"Accuracy | Test : {accuracy_score(np.sign(y_test), np.sign(model.predict(X_test)))}")

    print(f"MAE | Train : {mean_absolute_error(y_train, model.predict(X_train))}")
    print(f"MAE | Test : {mean_absolute_error(y_test, model.predict(X_test))}")
    
    print(f"MSE | Train : {mean_squared_error(y_train, model.predict(X_train))}")
    print(f"MSE | Test : {mean_squared_error(y_test, model.predict(X_test))}")


def manage_submission(model, name, scale = False):
    """ Make predictions from the model, process data before submission, save a csv on the current folder.

    Args:
        model (estimator object): Model used to make predictions
        name (str): Name to give to csv

    Returns:
        pd.DataFrame: Return a DataFrame with features of the submission set and predictions from the model
    """    
    
    X_eval = pd.read_csv("X_test_GgyECq8.csv")
    X_eval['DELIVERY_START'] = pd.to_datetime(X_eval['DELIVERY_START'], utc = True)
    X_eval['DELIVERY_START'] = X_eval['DELIVERY_START'].dt.tz_convert('Europe/Paris')
    X_eval = X_eval.set_index(['DELIVERY_START'])

    if scale : 
        scaler = StandardScaler()
        X_eval = scaler.fit_transform(X_eval)
        

    #X_eval = add_param(X_eval)
    
    X_eval = X_eval.drop(["predicted_spot_price"], axis = 1)

    col_na = X_eval['solar_power_forecasts_average']
    X_eval['solar_power_forecasts_average'] = col_na.fillna(col_na.mean())
    #X_eval.info()

    col_na = X_eval['solar_power_forecasts_std']
    X_eval['solar_power_forecasts_std'] = col_na.fillna(col_na.mean())
    #print(X_eval)
        
    y_eval = pd.Series(model.predict(X_eval), index = X_eval.index, name = 'spot_id_delta')
        
    y_eval.to_csv(f'submission_{name}.csv')
    
    return pd.concat([X_eval, y_eval], axis = 1)



def predict_spot_price(df_full, model):

    to_predict = df_full[df_full['predicted_spot_price'].isna()]
    pred = model.predict(to_predict.drop(['spot_id_delta', 'predicted_spot_price'], axis = 1))
    df_full.loc[df_full['predicted_spot_price'].isna(), 'predicted_spot_price'] = pred

    return df_full







def cross_grid(model, params, X, labels, weights):

    # Toutes les combinaisons de paramètres
    param_combinations = list(product(*params.values()))
    param_names = list(params.keys())
    
    #Définition du nombre de plis
    kf = KFold(n_splits = 5, shuffle=False)
    
    best_score = -np.inf
    best_params = None
    
    #Boucle sur les combinaisons
    for param in param_combinations :
        scores = []
        current_params = dict(zip(param_names, param))
        model = model.set_params(**current_params)
        
        # Boucle sur les plis
        for training_ids, val_ids in kf.split(X):
            training_set = X[training_ids]
            training_labels = labels[training_ids]
            training_weights = weights[training_ids]

            val_set = X[val_ids]
            val_labels = labels[val_ids]
            val_weights = weights[val_ids]
            
            #print(training_set)
            #print(training_labels)
            
            model.fit(training_set, training_labels)
            y_val_pred = model.predict(val_set)
            
            #scores.append(classif_score(val_labels, y_val_pred, val_weights))
        
        score = np.mean(scores)
        
        if score > best_score : 
            best_score = score
            best_params = current_params
            
        # Affichage des résultats pour la combinaison actuelle
        print(f"Params: {current_params} | Score moyen: {score:.4f}")
    
    # Affichage du meilleur résultat
    print(f"Best params: {best_params} | Best Score : {best_score:.4f}")

    return best_params