import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError,BooleanField
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect

from numpy import nan
import numpy as np

########################################
# Begin database stuff

# the connect function checks if there is a DATABASE_URL env var
# if it exists, it uses it to connect to a remote postgres db
# otherwise, it connects to a local sqlite db stored in predictions.db
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = TextField(unique=True)
    observation = TextField()
    outcome = BooleanField()
    true_outcome = BooleanField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)



DB2 = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')
class Requests(Model):
    request_type = TextField()
    observation = TextField()


    class Meta:
        database = DB2


DB2.create_tables([Requests], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################

# verifications 
def check_valid_column(observation):
    """
        Validates that our observation only has valid columns
        
        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    
    valid_columns = {
                    "observation_id",
                    "Type",
                    "Date",
                    "Part of a policing operation",
                    "Latitude",
                    "Longitude",
                    "Gender",
                    "Age range",
                    "Officer-defined ethnicity",
                    "Legislation",
                    "Object of search",
                    "station"
}

    keys = set(observation.keys())
    
    if len(valid_columns - keys) > 0: 
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        return False, error
    
    if len(keys - valid_columns) > 0: 
        extra = keys - valid_columns
        error = "Unrecognized columns provided: {}".format(extra)
        return False, error    

    return True, ""

def check_categorical_values(observation):
    """
        Validates that all categorical fields are in the observation and values are valid
        
        Returns:
        - assertion value: True if all provided categorical columns contain valid values, 
                           False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    
    valid_category_map = {
                'Type' : ['Person search', 'Person and Vehicle search', 'Vehicle search'] ,
                'Gender' : ['Male', 'Female', 'Other'] ,
                'Age range' : ['18-24', '25-34', 'over 34', '10-17', 'under 10'] ,
                'Officer-defined ethnicity' : ['Asian', 'White', 'Black', 'Other', 'Mixed'] ,
                'Object of search' : ['Controlled drugs', 'Offensive weapons', 'Stolen goods', 'Article for use in theft', 'Articles for use in criminal damage', 'Firearms', 'Anything to threaten or harm anyone', 'Crossbows', 'Evidence of offences under the Act', 'Fireworks', 'Psychoactive substances', 'Game or poaching equipment', 'Evidence of wildlife offences', 'Detailed object of search unavailable', 'Goods on which duty has not been paid etc.', 'Seals or hunting equipment'] ,
                'station' : ['devon-and-cornwall', 'dyfed-powys', 'derbyshire', 'bedfordshire', 'avon-and-somerset', 'cheshire', 'sussex', 'north-yorkshire', 'cleveland', 'merseyside', 'north-wales', 'wiltshire', 'norfolk', 'suffolk', 'thames-valley', 'durham', 'warwickshire', 'leicestershire', 'hertfordshire', 'cumbria', 'metropolitan', 'essex', 'south-yorkshire', 'surrey', 'staffordshire', 'northamptonshire', 'northumbria', 'city-of-london', 'nottinghamshire', 'gloucestershire', 'cambridgeshire', 'lincolnshire', 'btp', 'west-yorkshire', 'dorset', 'west-mercia', 'kent', 'hampshire', 'humberside', 'lancashire', 'greater-manchester', 'gwent'] 
                    }
    
    for key, valid_categories in valid_category_map.items():
        if key in observation:
            value = observation[key]
            if value not in valid_categories:
                error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                    key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
                return False, error
        else:
            error = "Categorical field {} missing"
            return False, error

    return True, ""


def check_Latitude(observation):
    """
        Validates that observation contains valid age value 
        
        Returns:
        - assertion value: True if age is valid, False otherwise
        - error message: empty if age is valid, False otherwise
    """
    
    value = observation.get("Latitude")
        

    if isinstance(value, str):
        error = "Field `Latitude` is not an float or a null value. Current value: " + str(value)
        return False, error

    return True, ""

def check_Longitude(observation):
    """
        Validates that observation contains valid age value 
        
        Returns:
        - assertion value: True if age is valid, False otherwise
        - error message: empty if age is valid, False otherwise
    """
    
    value = observation.get("Longitude")
        

    if isinstance(value, str):
        error = "Field `Longitude`  is not an float or a null value. Current value: " + str(value)
        return False, error

    return True, ""




########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/should_search/', methods=['POST'])
def predict():
    # flask provides a deserialization convenience function called
    # get_json that will work if the mimetype is application/json
    obs_dict = request.get_json()
    observation=obs_dict



    columns_ok, error = check_valid_column(observation)
    if not columns_ok:
        response = {'error': error}
        return jsonify(response)

    observation_data = Requests(
        request_type='should_search',
        observation=obs_dict
    )

    observation_data.save()
    DB2.commit()

    _id = obs_dict['observation_id']
    observation.pop('observation_id')


    categories_ok, error = check_categorical_values(observation)
    if not categories_ok:
        response = {'error': error}
        return jsonify(response)

    Latitude_ok, error = check_Latitude(observation)
    if not Latitude_ok:
        response = {'error': error}
        return jsonify(response)

    Longitude_ok, error = check_Longitude(observation)
    if not Longitude_ok:
        response = {'error': error}
        return jsonify(response)


    




    # # now do what we already learned in the notebooks about how to transform
    # # a single observation into a dataframe that will work with a pipeline
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    # # now get ourselves an actual prediction of the positive class
    proba = pipeline.predict_proba(obs)[0, 1]
    prediction = pipeline.predict(obs)[0]

    response = {  
                    'outcome': bool(prediction)
                }

    p = Prediction(
        observation_id=_id,
        outcome=bool(prediction),
        observation=obs_dict
    )
    try:
        p.save()

    except IntegrityError:
        error_msg = 'Observation ID: "{}" already exists'.format(_id)
        response['error'] = error_msg
        print(error_msg)
        DB.rollback()

    return jsonify(response)


@app.route('/search_result/', methods=['POST'])
def update():
    obs = request.get_json()

    observation_data = Requests(
        request_type='search_result',
        observation=obs
    )

    observation_data.save()
    DB2.commit()

    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.true_outcome = bool(obs['outcome'])
        p.save()
        DB.commit()

        values_on_db  = model_to_dict(p)
        values_on_db.pop('id')
        values_on_db.pop('observation')

        result = values_on_db
        result["predicted_outcome"] = result.pop("outcome")
        result["outcome"] = result.pop("true_outcome")

        return jsonify(result)
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['observation_id'])
        return jsonify({'error': error_msg})


@app.route('/list-db-contents', methods=['POST'])
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])


@app.route('/list-db-requests', methods=['POST'])
def list_db_requests():
    return jsonify([
        model_to_dict(obs) for obs in Requests.select()
    ])

# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(debug=True, port=5000)
