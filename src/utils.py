# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.dates as mpl_dates

# def compute_rmse(dataframe, column1, column2):
#     """
#     Computes the Root Mean Square Error (RMSE) between two columns in a pandas DataFrame.

#     Parameters:
#     dataframe (pandas.DataFrame): The DataFrame containing the columns for comparison.
#     column1 (str): Name of the first column.
#     column2 (str): Name of the second column.

#     Returns:
#     float: The Root Mean Square Error (RMSE) between the two columns.
#     """

#     # Ensure columns exist in the DataFrame
#     if column1 not in dataframe.columns or column2 not in dataframe.columns:
#         raise ValueError("One or both columns not found in the DataFrame.")

#     # Remove rows with missing values in either column
#     dataframe = dataframe.dropna(subset=[column1, column2])

#     # Compute RMSE
#     rmse = np.sqrt(np.mean((dataframe[column1] - dataframe[column2]) ** 2))
    
#     return rmse

# # # Assuming 'df' is your DataFrame and you want to compare 'Column_A' and 'Column_B':
# # rmse_value = compute_rmse(df, 'PM2_5', 'PM2.5')
# # print("RMSE:", rmse_value)


# def compute_NRMSE(RMSE, ref_column_data):
#     mean_of_ref = ref_column_data.mean()
#     NRMSE = (RMSE/mean_of_ref) * 100
#     return NRMSE

# # compute_NRMSE(rmse_value, df['PM2.5'])


# def plot_line_graph(dataframe, x_column, y_columns, labels):
#     sns.set_theme()
#     sns.set(style="whitegrid")

#     plt.figure(figsize=(15, 7))

#     for y_col, label in zip(y_columns, labels):
#         sns.lineplot(x=x_column, y=y_col, data=dataframe, label=label)

#     plt.gcf().autofmt_xdate()
#     date_format = mpl_dates.DateFormatter('%d/%m/%Y')
#     plt.gca().xaxis.set_major_formatter(date_format)

#     plt.title(f'{", ".join(labels)} vs Time')
#     plt.xlabel('Time')
#     plt.ylabel(f'{", ".join(labels)} Values')
#     plt.legend()

#     plt.show()
#     # plt.savefig('line_Plot_1_P.png', dpi=600)  # Save the plot if needed

# # # Assuming 'df' is your DataFrame
# # plot_line_graph(subset_df_copy, 'DataDate', ['PM2_5', 'PM2.5'], ['Sensor_ENE00950', 'Reference_Instrument'])


# def scatter_with_1to1_line(dataframe, x_column, y_column, color_column):
#     plt.figure(figsize=(10, 7))
    
#     # Scatter plot between x_column and y_column
#     plt.scatter(dataframe[x_column], dataframe[y_column], alpha=0.4, c=dataframe[color_column], cmap='jet')
    
#     # Adding a 1:1 line (diagonal line) to the scatter plot
#     plt.plot(dataframe[x_column], dataframe[x_column], 'r-', label='1:1 line')
    
#     # Labels, Legend, and Colorbar
#     plt.xlabel(x_column)
#     plt.ylabel(y_column)
#     plt.legend()
#     plt.colorbar(label=color_column)
#     plt.show()

# # # Assuming 'subset_df_copy' is your DataFrame
# # scatter_with_1to1_line(df,"PM2_5", "PM2.5", "RH")


# def resample_and_merge_csv(df1, df2):
# # %Y-%m-%d %H:%M:%S
#     # Convert the date columns to DataDate objects
#     df1['DataDate'] = pd.to_datetime(df1['DataDate'], format='%Y-%m-%d %H:%M:%S')
#     df2['DataDate'] = pd.to_datetime(df2['DataDate'], format='%d/%m/%Y %H:%M')

#     # Set the date column as the index for resampling
#     df1.set_index('DataDate', inplace=True)
#     df2.set_index('DataDate', inplace=True)

#     # Resample both DataFrames from seconds to minutes, using the mean
#     df1 = df1.resample('T').mean().dropna()
#     df2 = df2.resample('T').mean().dropna()

#     # Inner merge the two DataFrames based on the date index
#     merged_df = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
#     return merged_df, merged_df.isnull().sum()


# # merge_Teledyne_ENE00950, merged_csv_missing_values = resample_and_merge_csv(subset_data_ENE00950_copy, Teledyne)
# # print(merged_csv_missing_values)
# # merge_Teledyne_ENE00950.reset_index(inplace=True)
# # merge_Teledyne_ENE00950

# def calculate_hourly_std(dataframe, date_column, value_column):
#     dataframe[date_column] = pd.to_datetime(dataframe[date_column])
#     dataframe.set_index(date_column, inplace=True)
    
#     hourly_std = dataframe.groupby(pd.Grouper(freq='H')).apply(lambda x: x[value_column].std())
    
#     print("Hourly Standard Deviation:")
#     return hourly_std

# # result_hourly_std = calculate_hourly_std(data_ENE00950, 'DataDate', 'PM2_5')
# # (result_hourly_std[result_hourly_std <= 2])


# # Define the date parser function
# def custom_date_parser(date):
#     return pd.to_datetime(date, format='%d/%m/%Y %H:%M')  # Adjust the format according to your date format

# # # Read CSV file with dates and specify the date column and date parser function
# # data = pd.read_csv('your_file.csv', parse_dates=['DataDate'], date_parser=custom_date_parser)

# # # Display the first few rows of the DataFrame
# # print(data.head())


# # Define the date parser function
# def custom_date_parser_kunak(date):
# #     2023-10-06 00:04:18
#     return pd.to_datetime(date, format='%Y-%m-%d %H:%M:%S')  # Adjust the format according to your date format


# # Assuming 'df' is your DataFrame and 'DataDate' is the column containing dates
# # Replace 'df' and 'DataDate' with your actual DataFrame name and date column

# def filter_date_range(dataframe, start_date, end_date):
#     dataframe['DataDate'] = pd.to_datetime(dataframe['DataDate'])  # Convert to datetime if not already in datetime format
    
#     # Filter the DataFrame based on the date range
#     filtered_df = dataframe[(dataframe['DataDate'] >= start_date) & (dataframe['DataDate'] <= end_date)]
    
#     # Reset the index
#     filtered_df.reset_index(drop=True, inplace=True)
    
#     return filtered_df

# # # Example usage:
# # # Assuming 'df' is your DataFrame containing the 'DataDate' column
# # start_date = '2023-01-10 17:54:00'
# # end_date = '2023-12-11 23:59:00'

# # filtered_data = filter_date_range(df, start_date, end_date)
# # print(filtered_data)


# def resample(df1):
#     # Set the date column as the index for resampling
#     df1.set_index('DataDate', inplace=True)

#     # Resample both DataFrames from seconds to minutes, using the mean
#     df1 = df1.resample('D').mean().dropna()
    
#     return df1, df1.isnull().sum()

# # df, df_missing_values = resample(df)
# # print(df_missing_values)
# # df.reset_index(inplace=True)
# # df


# ##--------------------------------------------------------------------------

import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train,X_test,y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            # gs = GridSearchCV(model,para,cv=3)
            # gs.fit(X_train,y_train)

            # model.set_params(**gs.best_params_)
            # model.fit(X_train,y_train)

            model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)