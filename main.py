import pandas as pd
from Data_PreProcess import DataPreprocess
from Case_spesific_clean import ISKEYMA
from Forests import Random_Forest
from ExtremeGradientBoost import XGBoost



#file_name = "OMEGA BIG DATA.csv"

#Load_df = DataPreprocess(file_name)

#Preprocessed_data = Load_df.export_data()

### Case spesific cleaning
###---------------------------------------------------------------------------------------------------------------------
### Open all rows for the first training


#Case_spesific_cleaned = ISKEYMA(Preprocessed_data)
#Case_spesific_cleaned.save_data()

### Close all rows above this after first run if you want to save time

read_data = pd.read_csv("Test.csv", sep=";")
read_data = read_data.iloc[:, 1:]
loaded_data = pd.DataFrame(read_data)              #Output for the data



### Scale and 
### Train RandomForest Classifier
###---------------------------------------------------------------------------------------------------------------------

#Trees = Random_Forest(loaded_read_data)            #Input for the data

### Scale and 
### Train Extreme Gradient Boost
###---------------------------------------------------------------------------------------------------------------------

Gradient = XGBoost(loaded_data)                     #Input for the data


