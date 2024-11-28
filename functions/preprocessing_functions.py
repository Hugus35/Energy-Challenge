import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_hist_and_box(df, numeric_cols):
    """
    Plot histogram and boxplot for each feature of the DataFrame on the same line

    Args:
        df (pd.DataFrame): DataFrame containing features in column
        numeric_cols (list): Numerical columns to be plot
    """    
    for column in numeric_cols:
        plt.figure(figsize=(12, 4))  # Définir la taille de la figure
        
        # Sous-graphe 1 : Histogramme
        plt.subplot(1, 2, 1)  # 1 ligne, 2 colonnes, 1er plot
        sns.histplot(df[column], kde=True)  # Tracer un histogramme
        plt.title(f'Histogramme de {column}')
        
        # Sous-graphe 2 : Boxplot
        plt.subplot(1, 2, 2)  # 1 ligne, 2 colonnes, 2e plot
        sns.boxplot(x=df[column])  # Tracer un boxplot
        plt.title(f'Boxplot de {column}')
        
        # Afficher les deux graphiques
        plt.tight_layout()  # Ajuster pour éviter les chevauchements
        plt.show()


def detect_outliers_Z(df, column, plot = False, threshold = 3):
    """ Detect and delete outliers with a Z-test if delete = True. 

    Args:
        df (pd.DataFrame): DataFrame containing Dataset with features in column
        column (str): column to detect & delete outliers
        delete (bool, optional): if True, also deleting outliers in the DataFrame. Defaults to False.

    Returns:
        df (pd.DataFrame): Returning DataFrame modified or not according to delete parameters
    """    
    
    y = df[column]
    y.head()
    mu = y.mean()
    sigma = y.std()
    
    if plot :
        plt.figure()
        y.hist(bins = 200)
        plt.suptitle('Before deletion', fontsize=16)
    
    z = (y - mu)/sigma
    anomalies = [(i, y[i], z[i]) for i in range(len(y)) if np.abs(z[i]) > threshold]
    indices = [elt[0] for elt in anomalies]
    
    #print("Anomalies détectées (position, valeur, Z-score) : ", anomalies)
    print(f"{len(anomalies)} outliers for feature {column}")
    df = df.drop(df.index[indices])
    print(f"Number of dropped values : {len(indices)}")

    if plot :
        plt.figure()
        df[column].hist(bins = 200)
        plt.suptitle('After deletion', fontsize=16)

    return df


def detect_outliers_IQR(df, column, delete = False, threshold = 1.5):
   
    y = df[column]
    if delete :
        plt.figure()
        y.hist(bins = 200)
        plt.suptitle('Avant suppression', fontsize=16)

    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1

    anomalies = [(i, y[i]) for i in range(len(y)) if y[i] < Q1 - threshold*IQR or y[i] > Q3 + threshold*IQR]
    indices = [elt[0] for elt in anomalies]

    #print("Anomalies détectées (position, valeur) : ", anomalies)
    print(f"{len(anomalies)} outliers pour le feature {column}")

    if delete :
        df = df.drop(df.index[indices])
        plt.figure()
        df[column].hist(bins = 200)
        plt.suptitle('Après suppression', fontsize=16)
        print(f"Nombre de valeurs supprimées : {len(indices)}")

    return df


def delete_outliers_threshold(df, column, threshold_min, threshold_max):
    """ Delete outliers with with a simple threshold. 

    Args:
        df (pd.DataFrame): DataFrame containing Dataset with features in column
        column (str): column to detect & delete outliers
        threshold_min (int): drop values if below threshold_min
        threshold_max (int): drop values if above threshold_max


    Returns:
        df (pd.DataFrame): Returning DataFrame modified.
    """
        
    y = df[column]
    #y.head()
    mu = y.mean()
    sigma = y.std()
    
    plt.figure()
    y.hist(bins = 200)
    plt.suptitle('Before deletion', fontsize=16)
    mask = (y[column]<threshold_max) & (y[column]>threshold_min)
    y = y[mask]
    
    df[column] = y
    print(f"{np.sum(~mask)} outliers for feature {column}")
    
    plt.figure()
    df[column].hist(bins = 200)
    plt.suptitle('After deletion', fontsize=16)

    return df


def extract(X_url, y_url = None,  sep = ";"):
    """Load dataset, set index, delete columns and concat features and target variables

    Args:
        to_drop (list): list of columns to delete. If None, no deleted columns
        X_url (str) : string containing the url of X dataset
        y_url (str, optionnal) : string containing the url of y dataset

    Returns:
        df (pd.DataFrame): DataFrame resulting from the extraction
    """

    df = pd.read_csv(X_url, sep = sep)
    df['DELIVERY_START'] = pd.to_datetime(df['DELIVERY_START'], utc = True)
    df = df.set_index(['DELIVERY_START'])

    if y_url:
        y = pd.read_csv(y_url, sep = sep)
        y['DELIVERY_START'] = pd.to_datetime(y['DELIVERY_START'], utc = True)
        y = y.set_index(['DELIVERY_START'])
        df = pd.concat([df, y] , axis = 1)
    
    return df

def handle_missing(df, na = None, to_drop = None):
        
    if na == 'drop':
        df = df.dropna(subset = ['load_forecast', 'wind_power_forecasts_average', 'solar_power_forecasts_average', 
                                'coal_power_available'])
    if na == 'mean':
        col_na = df['load_forecast']
        df['load_forecast'] = col_na.fillna(col_na.mean())
       
        col_na = df['solar_power_forecasts_average']
        df['solar_power_forecasts_average'] = col_na.fillna(col_na.mean())
        
        col_na = df['solar_power_forecasts_std']
        df['solar_power_forecasts_std'] = col_na.fillna(col_na.mean())

        col_na = df['coal_power_available']
        df['coal_power_available'] = col_na.fillna(col_na.mean())
    
    if to_drop is not None:
        df = df.drop(to_drop, axis = 1)
    
    return df
                


def add_param(df):
    """extract new params from dates and coal_power_available

    Args:
        df (pd.DataFrame): Dataset

    Returns:
        pd.DataFrame: Dataset with new features
    """    
    
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    df["is_weekend"] = df.index.dayofweek > 5
    df["is_peak_hour"] = df.index.hour.isin(range(7, 19))

    df["is_weekend"] = df["is_weekend"]
    df["is_peak_hour"] = df["is_peak_hour"]
    
    df['gas_level'] = pd.cut(df['gas_power_available'], bins=[-np.inf, 10750, 11250, 11750, np.inf], labels=[0, 1, 2, 4])
    df['gas_level'] = df['gas_level'].astype(int)
    df = df.drop(['gas_power_available'], axis = 1)

    df['coal_level'] = pd.cut(df['coal_power_available'], bins=[-np.inf, 2500, 3000, np.inf], labels=[0, 1, 2])
    df['coal_level'] = df['coal_level'].astype(int)
    df = df.drop(['coal_power_available'], axis = 1)
    

    return df


def to_class(df):
    """Extract class from target variable (spot_id_delta) corresponding to its sign,
    adding it to the DataFrame

    Args:
        df (pd.DataFrame): Dataset

    Returns:
        df (pd.DataFrame): Dataset with new parameters
    """    
    
    df['class'] = None
    mask = df.loc[:, 'spot_id_delta'] >= 0
    
    df.loc[mask, 'class'] = 1
    df.loc[~mask, 'class'] = -1
    df['class'] = df['class'].astype(int)
    df = df.drop(['spot_id_delta'], axis = 1)
    
    return df


def compare_train_test(df, X_eval):

    for col in X_eval.columns:
        plt.figure()
        
        # Plot histograms for train and test
        sns.histplot(df[col], label='train', color='blue', kde=False, stat='density')
        sns.histplot(X_eval[col], label='test', color='orange', kde=False, stat='density')
        
        # Overlay KDE plots for train and test
        sns.kdeplot(df[col], color='blue', label='train KDE', linewidth=2)
        sns.kdeplot(X_eval[col], color='orange', label='test KDE', linewidth=2)
        
        plt.legend()
        plt.show()