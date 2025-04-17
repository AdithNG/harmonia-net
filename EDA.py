# Import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from the csv file into a pandas DataFrame
df = pd.read_csv("hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv")

# Remove the leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Print the table
print(df)   

# Seperate and section off the data by putting it in a nested dataframe
numeric_cols = [
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"]    
numeric_df = df[numeric_cols]
# Create a figure to visualise the correlation between the numerical and catagorical features of the dataset
plt.figure(figsize=(12, 10))
plt.title("Correlation Matrix")
correlation = numeric_df.corr()
# Create a colour/heat map 
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask we computed 
sns.heatmap(correlation, annot=True, fmt=".2f", linewidths=.5, cmap=cmap, square = True)

# Display the final figure
plt.show()

# Next we're going to further deconstruct the data in order to understand it better

# First we'll find the average for each genre in the dataset
mean_df = df.groupby('track_genre')[numeric_cols].mean()

# Sort the DataFrame by the mean value of the variable
mean_df = mean_df.sort_values(by='popularity', ascending=False)

# Create the plot
plt.figure(figsize=(6, 18))
sns.barplot(x='popularity', y=mean_df.index, data=mean_df)
plt.xticks(rotation=90)  

# Title and axis of the plot
plt.title('Average Popularity by Genre')
plt.xlabel('Popularity Index')
plt.ylabel('Genre')
plt.show()