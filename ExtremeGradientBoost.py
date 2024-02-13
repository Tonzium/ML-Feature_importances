import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from imblearn.over_sampling import SMOTE




class XGBoost:
    def __init__(self, data):
        self.data = data
        self.X, self.Y = self.Features_Labels()
        self.X_train_res, self.X_test_res, self.Y_train_res, self.Y_test_res = self.Split()
        self.X_train_scaled, self.X_test_scaled = self.Standard_scaler()
        self.model, self.predictions = self.XGBoost_model()
        self.Evaluation()
        self.Plot_feature_importances() 


    def Features_Labels(self):
        #Drop Wafer Number, do not need for training
        self.data = self.data.drop(columns=["remainder__Combined_WFR"])

        #Set all data to numerical to be able to calculate, but first change all , to .
        for col in self.data.columns:
            #Convert first all data to str to later convert it back
            self.data[col] = self.data[col].astype(str)

            if self.data[col].str.contains(",").any():
                self.data[col] = self.data[col].str.replace(",", ".").astype(float)

        #self.data = self.data.astype(float)

        # Features
        X = self.data.drop(columns=["remainder__ISKEYMA"]).astype(float)
        # Labels
        Y = self.data["remainder__ISKEYMA"].astype(int)


        return X, Y


    def Split(self):
        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, stratify=self.Y, test_size=0.2, random_state=2)

        # Using Smote overfitting because we are dealing with higly imbalanced data
        sampling_strat = {1: 1700}
        sm = SMOTE(sampling_strategy=sampling_strat, random_state=2)
        X_train_res, Y_train_res = sm.fit_resample(X_train, Y_train.ravel())
        X_test_res, Y_test_res = sm.fit_resample(X_test, Y_test.ravel())

        return X_train_res, X_test_res, Y_train_res, Y_test_res

        
    def Standard_scaler(self):
        scaler = StandardScaler()
        cols_to_scale = self.X_train_res.columns[1238:]

        self.X_train_res[cols_to_scale] = scaler.fit_transform(self.X_train_res[cols_to_scale])
        self.X_test_res[cols_to_scale] = scaler.fit_transform(self.X_test_res[cols_to_scale])

        return self.X_train_res, self.X_test_res


    def XGBoost_model(self):

        model = xgb.XGBClassifier(n_estimators=90, max_depth=5, learning_rate=0.05, random_state=1)

        model.fit(self.X_train_scaled, self.Y_train_res)

        predictions = model.predict(self.X_test_scaled)

        return model, predictions

    def Evaluation(self):

        accuracy = accuracy_score(self.Y_test_res, self.predictions)
        cm = confusion_matrix(self.Y_test_res, self.predictions)

        print("Confusion Matrix:")
        print(cm)
        print("")
        print("Correct answers:", cm[0,0]+cm[1,1])
        print("Incorrect answers:", cm[1,0]+cm[0,1])

        return print("Accuracy Score:", accuracy)
    
    
    def Plot_feature_importances(self):

        importances = self.model.feature_importances_
        feature_names = self.X.columns.tolist()

        # Create Dataframe
        feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        # Sort by importance
        sorted_feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
        
        topX = 50
        # Calculate Contribution of topX features -->
        Top_X_importance_features = feature_importance_df["Importance"].head(topX)
        # Sum of topX
        sum_topX_importances = Top_X_importance_features.sum()
        # Sum of all
        Total_importance = feature_importance_df["Importance"].sum()
        # Percentual contribution of topX
        total_contribution = sum_topX_importances / Total_importance
        print("TOP", topX ,"features contribution:", total_contribution)

        # Plot
        plt.figure(figsize=(15, 50))
        sns.barplot(x="Importance", y="Feature", data=sorted_feature_importance_df[:topX])
        plt.title("Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig("Feature_importances.png")