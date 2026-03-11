import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
 
#data loading
file_202409 = pd.read_csv("Project\\Data\\September_divvy_tripdata.csv")
file_202410 = pd.read_csv("Project\\Data\\October_divvy_tripdata.csv")
file_202411 = pd.read_csv("Project\\Data\\November_divvy_tripdata.csv")
file_202412 = pd.read_csv("Project\\Data\\December_divvy_tripdata.csv")
file_202501 = pd.read_csv("Project\\Data\\January 2025_divvy_tripdata.csv")
combined_data = pd.concat([file_202409, file_202410, file_202411, file_202412, file_202501], ignore_index=True)
 
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
if combined_data["ride_id"].isnull().sum() == 0 and combined_data["ride_id"].nunique() == len(combined_data):
    combined_data.set_index("ride_id", inplace=True)
combined_data["time_diff_sec"] = combined_data["end_time"] - combined_data["start_time"]
combined_data["ride_duration"] = combined_data["time_diff_sec"].dt.total_seconds() / 60
combined_data = combined_data.loc[combined_data["ride_duration"] > 0].reset_index(drop=True)
combined_data["date_week"] = pd.to_datetime(combined_data["start_date"]).dt.day_name() #(0=Monday, 6=Sunday)
 
#plotting time difference in minute
plt.figure(figsize=(8, 5))
plt.hist(combined_data["ride_duration"], bins=30, color="skyblue", edgecolor="black", alpha=0.7)
plt.axvline(combined_data["ride_duration"].median(), color='red', linestyle="--", label=f"Median: {combined_data["ride_duration"].median():.2f}")
plt.xlabel("Ride Duration (Minutes)")
plt.ylabel("Frequency")
plt.title("Histogram of Ride Duration in Minutes")
plt.legend()
plt.show()
 
#idenitfying and dealing with outliers
Q1 = combined_data["ride_duration"].quantile(0.25)
Q3 = combined_data["ride_duration"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
median = combined_data["ride_duration"].median()
outliers = combined_data[(combined_data["ride_duration"] < lower_bound) | (combined_data["ride_duration"] > upper_bound)]
print(outliers.sort_values(by="ride_duration", ascending=True))
combined_data = combined_data[(combined_data["ride_duration"] < upper_bound) & (combined_data["ride_duration"] > lower_bound)]
print(combined_data.sort_values(by="ride_duration", ascending=False))
print(combined_data.info())
print(combined_data.describe())
print("Start Station ID:\n", combined_data["start_station_id"].nunique())
print("Start Station Name:\n", combined_data["start_station_name"].nunique())
print("End Station ID:\n", combined_data["end_station_id"].nunique())
print("End Station Name:\n", combined_data["end_station_name"].nunique())
 
#plotting outliers in boxplot
plt.figure(figsize=(6, 4))
ax = sns.boxplot(y=combined_data["ride_duration"], color="lightblue")
ax.text(0, Q1, f"Q1: {Q1:.2f}", ha="left", va="center", fontsize=10, color="blue")
ax.text(0, Q3, f"Q3: {Q3:.2f}", ha="left", va="center", fontsize=10, color="blue")
ax.text(0, median, f"Median: {median:.2f}", ha="left", va="center", fontsize=10, color="green")
ax.text(0, lower_bound, f"Lower Bound: {lower_bound:.2f}", ha="right", va="center", fontsize=10, color="red")
ax.text(0, upper_bound, f"Upper Bound: {upper_bound:.2f}", ha="right", va="center", fontsize=10, color="red")
plt.title("Boxplot for Outliers in Time Differance")
plt.ylabel("ride_duration")
plt.show()
 
#splitting train test and unseen data from main dataframe
date_threshold = dt.strptime("2024-12-31", "%Y-%m-%d").date()
combined_data_train_test = combined_data[combined_data["start_date"] <= date_threshold].reset_index(drop=True)
combined_data_unseen = combined_data[combined_data["start_date"] > date_threshold].reset_index(drop=True)

# Rideable Type Distribution
plt.figure(figsize=(8,6))
ax=sns.countplot(x='rideable_type', data=combined_data)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=10, fmt='%d', padding=3)
plt.title('Distribution of Rideable Types')
plt.xlabel('Rideable Type')
plt.ylabel('Number of Rides')
plt.show()


# Membership Type Distribution
plt.figure(figsize=(6,6))
ax=sns.countplot(x='member_casual', data=combined_data)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=10, fmt='%d', padding=3)
plt.title('Distribution of Membership Status')
plt.xlabel('Membership Status')
plt.ylabel('Number of Rides')
plt.show()

# Ride Duration by Member Type
plt.figure(figsize=(10,6))
ax=sns.boxplot(x='member_casual', y='ride_duration', data=combined_data)
for i, member_type in enumerate(combined_data['member_casual'].unique()):
    subset = combined_data[combined_data['member_casual'] == member_type]['ride_duration']
    
    Q1 = np.percentile(subset, 25)  # First quartile
    Q3 = np.percentile(subset, 75)  # Third quartile
    Median = np.median(subset)      # Median
    IQR = Q3 - Q1                   # Interquartile range
    Lower_Bound = Q1 - 1.5 * IQR
    Upper_Bound = Q3 + 1.5 * IQR
    
    # Annotate values on the boxplot
    ax.text(i, Q1, f'Q1: {Q1:.1f}', ha='center', va='bottom', fontsize=10, color='blue')
    ax.text(i, Q3, f'Q3: {Q3:.1f}', ha='center', va='top', fontsize=10, color='blue')
    ax.text(i, Median, f'Median: {Median:.1f}', ha='center', va='center', fontsize=10, fontweight='bold', color='red')
    ax.text(i, Lower_Bound, f'LB: {Lower_Bound:.1f}', ha='center', va='bottom', fontsize=10, color='green')
    ax.text(i, Upper_Bound, f'UB: {Upper_Bound:.1f}', ha='center', va='top', fontsize=10, color='green')
plt.title('Ride Duration by Member Type')
plt.xlabel('Member Type')
plt.ylabel('Ride Duration (Minutes)')
plt.show()

# Create a crosstab of 'day_of_week' and 'member_casual'
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
combined_data['date_week'] = pd.Categorical(combined_data['date_week'], categories=day_order, ordered=True)
rides_by_day = pd.crosstab(combined_data['date_week'], combined_data['member_casual'])

plt.figure(figsize=(10, 6))
ax = rides_by_day.plot(kind='bar', color=['skyblue', 'lightgreen'], width=0.8)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=10, fmt='%d', padding=3)

# Adding titles and labels

plt.title('Trips by Day of the Week: Member vs Casual', fontsize=14)
plt.xlabel('Day of the Week', fontsize=12)
plt.ylabel('Number of Rides', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Membership Status')
plt.tight_layout()
plt.show()

#Remove 'August' from the data
combined_data['month'] = pd.to_datetime(combined_data['start_date']).dt.month_name()
 
# Drop August from the dataset
combined_data = combined_data[combined_data['month'] != 'August']
 
# Set the correct order for months
month_order = ["September", "October", "November", "December", "January"]
 
# Convert 'month' to a categorical column with the correct order
combined_data['month'] = pd.Categorical(combined_data['month'], categories=month_order, ordered=True)
 
# Create a crosstab of 'month' and 'member_casual'
rides_by_month = pd.crosstab(combined_data['month'], combined_data['member_casual'])
 
# Plotting the line graph
plt.figure(figsize=(10, 6))
ax = rides_by_month.plot(kind='line', marker='o', color=['skyblue', 'lightgreen'], linewidth=2)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=10, fmt='%d', padding=3)
# Adding labels to each data point
for line in ax.lines:
    for x, y in zip(line.get_xdata(), line.get_ydata()):
        ax.text(x, y, f'{int(y)}', color='black', ha='center', va='bottom', fontsize=10)
 
# Adding titles and labels
plt.title('Trips by Month: Member vs Casual (Excluding August)', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Rides', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Membership Status')
plt.tight_layout()
 
# Show the plot
plt.show()

# Count rides by start station
plt.figure(figsize=(12,6))
ax=combined_data['start_station_name'].value_counts().head(20).plot(kind='bar')
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=10, fmt='%d', padding=3)
plt.title('Top 20 Start Stations')
plt.ylabel('Number of Rides')
plt.xlabel('Station Name')
plt.show()

# Top 20 End Stations
plt.figure(figsize=(12,6))
ax=combined_data['end_station_name'].value_counts().head(20).plot(kind='bar', color='lightgreen')
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=10, fmt='%d', padding=3)
plt.title('Top 20 End Stations')
plt.xlabel('Station Name')
plt.ylabel('Number of Rides')
plt.xticks(rotation=45)
plt.show()

# Count rides by start station and membership status
rides_by_station_member_casual = pd.crosstab(combined_data['start_station_name'], combined_data['member_casual'])
 
# Get the top 10 stations by total number of rides (sum of both member and casual)
top_stations = rides_by_station_member_casual.sum(axis=1).sort_values(ascending=False).head(10)
 
# Filter the data to only include the top 10 stations
rides_by_top_stations = rides_by_station_member_casual.loc[top_stations.index]
 
# Plotting the grouped bar chart
plt.figure(figsize=(12, 6))
ax=rides_by_top_stations.plot(kind='bar', width=0.8)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=10, fmt='%d', padding=3)
# Adding titles and labels
plt.title('Top 10 Start Stations: Member vs Casual', fontsize=14)
plt.xlabel('Station Name', fontsize=12)
plt.ylabel('Number of Rides', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Membership Status', labels=['Casual', 'Member'])
plt.tight_layout()
 
# Show the plot
plt.show()

# Count rides by start station and membership status
rides_by_station_member_casual = pd.crosstab(combined_data['end_station_name'], combined_data['member_casual'])
 
# Get the top 10 stations by total number of rides (sum of both member and casual)
top_stations = rides_by_station_member_casual.sum(axis=1).sort_values(ascending=False).head(10)
 
# Filter the data to only include the top 10 stations
rides_by_top_stations = rides_by_station_member_casual.loc[top_stations.index]
 
# Plotting the grouped bar chart
plt.figure(figsize=(12, 6))
ax=rides_by_top_stations.plot(kind='bar', width=0.8)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=10, fmt='%d', padding=3)
 
# Adding titles and labels
plt.title('Top 10 End Stations: Member vs Casual', fontsize=14)
plt.xlabel('Station Name', fontsize=12)
plt.ylabel('Number of Rides', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Membership Status', labels=['Casual', 'Member'])
plt.tight_layout()
 
# Show the plot
plt.show()

# Top 10 Start Stations: Rideable Type
rides_by_station_rideable_type = pd.crosstab(combined_data['start_station_name'], combined_data['rideable_type'])
top_stations = rides_by_station_rideable_type.sum(axis=1).sort_values(ascending=False).head(10)
rides_by_top_stations = rides_by_station_rideable_type.loc[top_stations.index]
ax=rides_by_top_stations.plot(kind='bar', figsize=(12, 6), width=0.8)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=10, fmt='%d', padding=3)
plt.title('Top 10 Start Stations by Rideable Type')
plt.xlabel('Station Name')
plt.ylabel('Number of Rides')
plt.legend(title='Rideable Type')
plt.xticks(rotation=45)
plt.show()
plt.show()

# Top 10 End Stations: Rideable Type
rides_by_station_rideable_type = pd.crosstab(combined_data['end_station_name'], combined_data['rideable_type'])
top_stations = rides_by_station_rideable_type.sum(axis=1).sort_values(ascending=False).head(10)
rides_by_top_stations = rides_by_station_rideable_type.loc[top_stations.index]
ax=rides_by_top_stations.plot(kind='bar', figsize=(12, 6), width=0.8)
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=10, fmt='%d', padding=3)
plt.title('Top 10 End Stations by Rideable Type')
plt.xlabel('Station Name')
plt.ylabel('Number of Rides')
plt.legend(title='Rideable Type')
plt.xticks(rotation=45)
plt.show()
