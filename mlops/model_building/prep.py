# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
#For converting text data into numerical representation
from sklearn.preprocessing import LabelEncoder
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Pavan-K492/tourism-package-prediction/tourism.csv"
bank_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'CityTier',                    # The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3).
    'NumberOfPersonVisiting',      # Total number of people accompanying the customer on the trip.
    'Passport',                    # Whether the customer holds a valid passport (0: No, 1: Yes).
    'PitchSatisfactionScore',      # Score indicating the customer's satisfaction with the sales pitch.
    'OwnCar',                      # Whether the customer owns a car (0: No, 1: Yes).
    'Age',                         # Age of the customer.
    'DurationOfPitch',             # Duration of the sales pitch delivered to the customer.
    'NumberOfFollowups',           # Total number of follow-ups by the salesperson after the sales pitch.-
    'PreferredPropertyStar',       # Preferred hotel rating by the customer.
    'NumberOfTrips',               # Average number of trips the customer takes annually.
    'NumberOfChildrenVisiting',    # Number of children below age 5 accompanying the customer
    'MonthlyIncome'                # Gross monthly income of the customer.
]


#dropping 1st column which is unnamed or index column and CustomerID which has unique identifier from modelling as it not useful for preprocessing
#data.drop(columns=[data.columns[0], 'CustomerID'], inplace=True)

#Labeling the categorical features for faster processing
label_encoder = LabelEncoder()
data['TypeofContact'] = label_encoder.fit_transform(data['TypeofContact'])
data['Occupation'] = label_encoder.fit_transform(data['Occupation'])
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['ProductPitched'] = label_encoder.fit_transform(data['ProductPitched'])
data['MaritalStatus'] = label_encoder.fit_transform(data['MaritalStatus'])
data['Designation'] = label_encoder.fit_transform(data['Designation'])

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',         # The method by which the customer was contacted (Company Invited or Self Inquiry).
    'Occupation',            # Customer's occupation (e.g., Salaried, Freelancer).
    'Gender',                # Gender of the customer (Male, Female).
    'ProductPitched',        # The type of product pitched to the customer.
    'MaritalStatus',         # Marital status of the customer (Single, Married, Divorced).
    'Designation',           # Customer's designation in their current organization.
]

# Define predictor matrix (X) using selected numeric and categorical features
X = data[numeric_features + categorical_features]

# Define target variable
y = data[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="Pavan-K492/tourism-package-prediction",
        repo_type="dataset",
    )
