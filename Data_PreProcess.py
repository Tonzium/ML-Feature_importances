
import pandas as pd


class DataPreprocess:
    def __init__(self, file_name):
        self.file_name = file_name
        self.df = self.extract_data()
        self.df = self.Include_BA9_data_only()
        self.df = self.Include_PRODA_data_only()
        self.df = self.remove_null_rows()
        self.df = self.remove_XALL_WAVI_cols()
        self.df = self.remove_cols()
        self.df = self.feature_engineer()

    def extract_data(self):

        self.df = pd.read_csv(self.file_name)
        return self.df
    
    def data_summary(self):

        print(self.df.head(5))

        print(self.df.columns)

        print(self.df.info)

        #list_of_cols = list(self.df.columns)
        #small_list_of_cols = list_of_cols[0]

        #list_of_data = []

        #for i in list_of_cols:
        #    list_of_data.append(self.df[i].isnull().sum())

        #print(list_of_data)


    def data_dtype(self, column_name):

        print(f"Column", {column_name}, "dtype is:")

        print(self.df[column_name].dtypes)



    def Include_BA9_data_only(self):

        self.df = self.df[self.df["Comb...."].str.startswith('BA9')] # cencored col

        return self.df

    def Include_PRODA_data_only(self):

        self.df = self.df[self.df["EXP..."] == "0"] # cencored col

        return self.df
    
    def remove_null_rows(self):

        self.df = self.df[self.df["TOTALYIELD AC....."].notnull()] # cencored col

        return self.df

    def remove_XALL_WAVI_cols(self):

        remove_list_of_cols10 = ["XALL WAVI_..."] #censored the list of cols

        self.df.drop(remove_list_of_cols10, axis=1, inplace=True)

        return self.df
    
    def remove_cols(self):

        remove_list_of_cols11 = ["STDDEVALL WAVI_PRO..."] # cencored the list of cols
        
        remove_another_list_of_cols11 = ["STDDEVALL WAVI_PR.."] # cencored the list of cols
        
        remove_list_of_cols12 = ["EXT02 FC 720...."] # cencored the list of cols
        
        remove_list_of_cols13 = ["STDDEVALL EX..."] # cencored the list of cols

        self.df.drop(remove_list_of_cols11, axis=1, inplace=True)
        self.df.drop(remove_another_list_of_cols11, axis=1, inplace=True)
        self.df.drop(remove_list_of_cols12, axis=1, inplace=True)
        self.df.drop(remove_list_of_cols13, axis=1, inplace=True)

        return self.df
    
    def feature_engineer(self):

        # If certain string set 1 else 0

        self.df["EQP_C...."] = self.df["EQP_CO..."].apply(lambda x: 1 if x == "PNT_06" else 0) # cencored col

        self.df["EQP_T..."] = self.df["EQP_T...."].apply(lambda x: 1 if x == "MITTADATA-LASKENTA" else 0) # cencored col

        return self.df


    def save_data(self):

        save_file_name = "Test.csv"
        
        self.df.to_csv(save_file_name, sep=";")

        print("Data saved as", save_file_name)

    def export_data(self):
        return self.df