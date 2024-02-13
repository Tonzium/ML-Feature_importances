import pandas as pd
from sklearn.compose import ColumnTransformer
from category_encoders import BinaryEncoder




class ISKEYMA:
    def __init__(self, data):
        self.data = data
        self.data = self.clean_THIN_wafer_data()
        self.data = self.clean_COMB_wafer_data()
        self.data = self.notes_col()
        self.data = self.bad_wafers()
        self.data = self.clean_other()
        self.data = self.dummies()
        self.data = self.remove_null_rows()
        



# ISKEYMA problem does not relate to THN wafers
    def clean_THIN_wafer_data(self):
        
        remove_THIN_cols = self.data.filter(regex='^(EQP..|DAY...|OPER...)') #cencored col

        self.data = self.data.drop(columns=remove_THIN_cols)

        self.data = self.data.drop(columns="TH..") #cencored col

        return self.data

# ISKEYMA problem does not relate to COMBINED wafers
    def clean_COMB_wafer_data(self):
        
        remove_THIN_cols = self.data.filter(regex='^(EQP..|DAY..|OPER...)') #cencored col

        self.data = self.data.drop(columns=remove_THIN_cols)

        return self.data
    
    def notes_col(self):
        
        self.data["NOTES"] = self.data["NOTES"].apply(lambda x: 1 if pd.notnull(x) and x != "" else 0)

        return self.data

    def bad_wafers(self):

        bad_wafers = ["BA9I...."]   # censored the list of wafers

        # Testing if clear correlation would be found how much would be the importance coefficient. Figure name: Feature_importances_with_perfect_col.png
        # Importance coefficient is high ~0.1 for single feature !
        #self.data["ISKEY..."] = self.data["Comb..."].isin(bad_wafers).astype(int)

        self.data["ISKEY.."] = self.data["Comb..."].isin(bad_wafers).astype(int) #cencored cols

        return self.data


#This might need to be adjusted
    def clean_other(self):
        
        # THICKWFR will be running number and it can be tracked from index
        mass_filter = ["THICK..."] #cencored the list of cols

        self.data = self.data.drop(columns=mass_filter)

        #Also remove timestamps for now
        filter_timestamps = self.data.filter(regex='^(DAY_THK)').columns

        self.data = self.data.drop(columns=filter_timestamps)

        return self.data
        


    def dummies(self):

        # add null to all empty values because we want to categorize null values
        data_added_nulls = self.data.fillna("nan")

        # Number 1 split

        categorical_features = self.data.filter(regex='^(EQP...|OPE...|SUPP..)').columns
        
        ct = ColumnTransformer(
            [("binary_encoder", BinaryEncoder(), categorical_features)],
             remainder="passthrough"
        ) #remainder includes other than categorical data into table

        data_transformed = ct.fit_transform(data_added_nulls)

        feature_names = ct.get_feature_names_out()

        data_formed_1 = pd.DataFrame(data_transformed, columns=feature_names)

        return data_formed_1

    def remove_null_rows(self):

        # Remaining null values can be removed
        self.data = self.data.replace("nan", pd.NA)
        self.data = self.data.dropna(how="any")
        
        return self.data

    def save_data(self):

        save_file_name = "Test.csv"
        
        self.data.to_csv(save_file_name, sep=";")

        print("Data saved as", save_file_name)