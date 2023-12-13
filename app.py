from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging
from src.utils import resample_and_merge_csv


application = Flask(__name__)
app = application

# Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for calibration 
@app.route('/calibration-engine-api/PM25/v1/', methods=['GET', 'POST'])
def predict_datapoints():
    if request.method == 'GET':
        make_predictions = {'Do the prediction' : 'Do the prediction'}
        return jsonify(make_predictions)
    
    else:
        # Load JSON string and convert it to a Pandas DataFrame
        logging.info('Load JSON string and convert it to a Pandas DataFrame')
        json_data = request.json
        df = pd.read_json(json_data)
        df['DataDate'] = df['DataDate'].apply(convert_to_datetime)  
        print(df)

        # Filter Rows
        logging.info('Filter Rows')
        filtered_df = filter_data(df)
        filtered_df_copy = filtered_df.copy()
        print(filtered_df_copy)

        # Convert 'DataDate' column to datetime
        logging.info("Convert 'DataDate' column to datetime")
        filtered_df['DataDate'] = pd.to_datetime(filtered_df['DataDate'])

        # Engineer columns
        logging.info("Engineer columns")
        processed_df = preprocess_data(filtered_df)
        logging.info(processed_df)
        print(processed_df)

        #Send columns to the model
        # logging.info('Send these columns to the model: PM2_5', 'PM_10', 'RH', 'Temp', 'PM2_5-PM10', 'Month', 'Hour')
        pred_df = processed_df[['PM2_5', 'PM_10', 'RH', 'Temp', 'PM2_5-PM10', 'Month', 'Hour']]

        #make predictions
        logging.info('Make prediction')
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Convert NumPy array to DataFrame
        predicted_df = pd.DataFrame(results, columns=['calibrated_PM2_5'], index = filtered_df_copy.index)
        logging.info(predicted_df)
        print(predicted_df)

        #attach predicted_df  to df_copy
        # Concatenate along rows (axis=0)
        combined_df = pd.concat([filtered_df_copy, predicted_df], axis=1)
        logging.info(combined_df)
        print(combined_df)

        # Convert DataFrame to JSON
        logging.info("Convert DataFrame to JSON")
        json_data_converted = combined_df.to_json(orient='records')
        logging.info('Dataframe successfully converted to Json Data')

        # print(json_data_converted)
        return json_data_converted
       
if __name__=="__main__":
    app.run(host="0.0.0.0", debug = True)        
