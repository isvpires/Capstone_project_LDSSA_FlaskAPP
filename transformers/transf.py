
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class Mask_OutcomePositive(BaseEstimator, TransformerMixin):

    #Return self nothing else to do here
    def fit( self, X=None, y = None):
        return self
    #Method that describes what we need this transformer to do
    def transform( self, X, y=None):

        df = X.copy()
        def mask(value):
            if value == 'A no further action disposal':
                return False
            elif value == 'Nothing found - no further action':
                return False
            else: 
                return True

        df['outcome_positive'] = df.Outcome.apply(mask)
        return df



class Mask_SearchSuccess(BaseEstimator, TransformerMixin):

    #Return self nothing else to do here
    def fit( self, X, y = None):
        return self
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None):

        df = X.copy()
        def mask(a, b):
            if a == True and b == True:
                return True
            else:
                return False  

        df['search_success'] = df.apply(lambda row: mask(row['outcome_positive'], row['Outcome linked to object of search']), axis=1)
        return df

class Filter_ColumnValue(BaseEstimator, TransformerMixin):
    """

    """
    def __init__(self, col, value):
        self.col = col
        self.value = value

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X = X[X['station'] != self.value]
        
        return X

class FilterRowsYear(BaseEstimator, TransformerMixin):
    """

    """
    def __init__(self, date_feature, year):
        self.date_feature = date_feature
        self.year = year

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X[self.date_feature] = pd.to_datetime(X[self.date_feature], errors='coerce', yearfirst=True, format="%Y-%m-%dT%H:%M:%S+%f:00")
        X = X[ X[self.date_feature].dt.year == self.year]

        return X

class FilterRowsThreshold(BaseEstimator, TransformerMixin):
    """

    """
    def __init__(self, col, threshold):
        self.col = col
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        filtered_list = X_copy.groupby(self.col)['observation_id'].count() < len(df)*self.threshold
        filtered_list = list(filtered_list[filtered_list.values == True].keys())
        X = X[X[self.col].isin(filtered_list)]
        return X

class FixNA_OutcomeLinkedSearch(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X_, y=None):
        X = X_.copy()

        #stations without location known - btp   
        X['Outcome linked to object of search'] = X['Outcome linked to object of search'].fillna(False)


        return X

class FixNA_Coordinates(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X_, y=None):
        return self

    def transform(self, X_, y=None):
        X = X_.copy()
        columns = X.columns

        feature_name = 'station'
        stations_location = pd.read_csv('data/stations_location_2019.csv', sep=',')
        stations_location = stations_location[['PFA19NM', 'LONG', 'LAT','Shape__Area']]
        stations_location = stations_location.rename(columns={'PFA19NM':'station'})
        stations_location[feature_name] = stations_location[feature_name] \
                                                .str.lower() \
                                                .str.replace('police','') \
                                                .str.replace('&','and') \
                                                .str.replace('london, city of','city of london') \
                                                .str.strip() \
                                                .str.replace(' ','-')

        X = X.merge(stations_location,how='left', left_on='station', right_on='station')

        #stations known   
        X["Latitude"] = X["Latitude"].fillna(X["LAT"])
        X["Longitude"] = X["Longitude"].fillna(X["LONG"])

        #stations without location known - btp   
        X["Latitude"] = X["Latitude"].fillna(X["Latitude"].median())
        X["Longitude"] = X["Longitude"].fillna(X["LONG"].median())


        return X[columns]

class FixNA_Selfdefinedethnicity(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X_, y=None):
        data = X_.copy()
        data_ethnicity  = data.groupby(["Officer-defined ethnicity","Self-defined ethnicity"])['observation_id'].count().sort_values().reset_index()
        self.mapping = data_ethnicity.set_index(["Officer-defined ethnicity"])["Self-defined ethnicity"].to_dict()

        return self
    def transform(self, X_, y=None):
        X = X_.copy()
        X["Self-defined ethnicity"] = X.apply(lambda row: self.mapping[row["Officer-defined ethnicity"]]
                                if (pd.isnull(row["Self-defined ethnicity"]) and row["Officer-defined ethnicity"] in self.mapping.keys())
                                else row["Self-defined ethnicity"], axis=1)
        return X

class DateTransformer(BaseEstimator, TransformerMixin):
    """
    ok
    """
    def __init__(self, date_feature='Date'):
        self.date_feature = date_feature
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):

        df =  X.copy()
        def clean_df_time(df_):
            # fill null values with average from other wind towers
            df = df_.copy()
            df['Date'] = pd.to_datetime(df[self.date_feature], errors='coerce',yearfirst=True, format="%Y-%m-%dT%H:%M:%S+%f:00")# ,  infer_datetime_format=True, ,format="%Y-%m-%dT%H:%M:%S-%f"
            

            return df 
        

        df = clean_df_time(df)


        df['Day'] = df['Date'].dt.date
        df['Hour'] = df['Date'].dt.hour
        df['Minute'] = df['Date'].dt.minute


        df['Time (sin)'] = np.sin((2. * (df['Hour']+ df['Minute']/60) *  np.pi / 24)) 
        df['Time (cos)'] = np.sin((2. * (df['Hour']+ df['Minute']/60) *  np.pi / 24)) 

        return df