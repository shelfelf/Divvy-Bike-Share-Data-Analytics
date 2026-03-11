
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datetime import timedelta

#Use rideable type & member/casual to classify or cluster and identify decisions about start/end station and start/end time
#KNN - Harshil
#Random Forest - Darshan
#XGBoost - Vivek
#Reporting and PPT - Ananya & Sakshi

#data loading
file_202409 = pd.read_csv("Data\\September_divvy_tripdata.csv")
file_202410 = pd.read_csv("Data\\October_divvy_tripdata.csv")
file_202411 = pd.read_csv("Data\\November_divvy_tripdata.csv")
file_202412 = pd.read_csv("Data\\December_divvy_tripdata.csv")
combined_data = pd.concat([file_202409, file_202410, file_202411, file_202412], ignore_index=True)

#data cleaning and preparation
combined_data.isnull().sum()
combined_data.dropna(inplace=True)
combined_data["start_date"] = pd.to_datetime(combined_data["started_at"]).dt.date
combined_data["end_date"] = pd.to_datetime(combined_data["ended_at"]).dt.date
combined_data["start_time"] = pd.to_datetime(combined_data["started_at"])
combined_data["end_time"] = pd.to_datetime(combined_data["ended_at"])
combined_data = combined_data.drop(columns=["started_at", "ended_at", "start_lat", "start_lng", "end_lat", "end_lng"]).reset_index(drop=True)
combined_data["start_day"] = pd.to_datetime(combined_data["start_date"]).dt.day_name()
combined_data["end_day"] = pd.to_datetime(combined_data["end_date"]).dt.day_name()
combined_data.set_index("ride_id", inplace=True)
combined_data["time_diff"] = combined_data["end_time"] - combined_data["start_time"]
combined_data["time_min"] = combined_data["time_diff"].dt.total_seconds() / 60
combined_data = combined_data.where(combined_data["time_min"] > 0).reset_index(drop=True)

#idenitfying and dealing with outliers
Q1 = combined_data["time_min"].quantile(0.25)
Q3 = combined_data["time_min"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = combined_data[(combined_data["time_min"] < lower_bound) | (combined_data["time_min"] > upper_bound)]
print(outliers.sort_values(by="time_min", ascending=False))
combined_data = combined_data[(combined_data["time_min"] >= lower_bound) & (combined_data["time_min"] < upper_bound)]
print(combined_data.sort_values(by="time_min", ascending=False))