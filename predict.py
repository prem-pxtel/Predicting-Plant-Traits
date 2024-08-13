import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime

'''
Using CNNs and Hyperparameter-Optimized Boosting Techniques to Predict Vital Plant Traits
Prem Patel

To run the prediction pipeline, simply download predict.py and run the script in a terminal or PowerShell window. 
Enter 'G', 'B', or 'N' based on the desired tuning method (G for Grid search, B for Bayesian search, or N for no tuning - fastest). 
This is the only required input; final predictions will be saved as a .csv file.

See report.pdf for a full breakdown of the project.
'''

def get_tuning_flag():
    valid_flags = ['G', 'B', 'N']
    while True:
        tuning_flag = input('Enter tuning method: G for GridSearch, B for Bayesian Search, N for None: ').strip().upper()
        if tuning_flag in valid_flags:
            return tuning_flag
        else:
            print("Invalid input. Please enter 'G', 'B', or 'N'.")

# Data preprocessing functions
def remove_outliers(data, threshold=3, inlier_percent=0.8):
    ids = data.iloc[:, 0]
    features = data.iloc[:, 1:]
    z_scores = np.abs((features - features.mean()) / features.std())
    min_features_inlier = int(inlier_percent * features.shape[1])
    row_outlier_mask = (z_scores < threshold).sum(axis=1) >= min_features_inlier
    filtered_data = features[row_outlier_mask]
    filtered_ids = ids[filtered_data.index]
    filtered_data = pd.concat([filtered_ids, filtered_data], axis=1)
    # filtered_data.to_csv('testing/filtered_data.csv', index=False)
    return filtered_data

def normalize(df, is_train=True, min_max_norm=True):
    global norm_param1, norm_param2
    ids = df.iloc[:, 0]
    data = df.iloc[:, 1:]
    if is_train:
        if min_max_norm:
            norm_param1 = data.min()
            norm_param2 = data.max()
        else:
            norm_param1 = data.mean()
            norm_param2 = data.std()
    data_normalized = data.copy()
    if min_max_norm:
        for column in data.columns:
            data_normalized[column] = (data[column] - norm_param1[column]) / (norm_param2[column] - norm_param1[column])
    else:
        for column in data.columns:
            data_normalized[column] = (data[column] - norm_param1[column]) / (norm_param2[column])
    data_normalized = pd.concat([ids, data_normalized], axis=1)
    # data_normalized.to_csv(f'testing/processed_data{is_train}.csv', index=False)
    return data_normalized

def preprocess_data(csv_file, is_train=True):
    data = pd.read_csv(csv_file)
    if is_train:
        data = remove_outliers(data)
    data.iloc[:, -6:] = np.log1p(data.iloc[:, -6:])
    data = normalize(data, is_train)
    return data

def random_color_jitter(img, probability=0.5):
    if torch.rand(1).item() < probability:
        color_jitter = transforms.ColorJitter(
            contrast=(0.9, 1.1), 
            saturation=(0.9, 1.1), 
            brightness=(0.9, 1.1)
        )
        img = color_jitter(img)
    return img

class PlantDataset(Dataset):
    def __init__(self, images_folder, data, transform=None):
        self.images_folder = images_folder
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = f"{self.images_folder}/{self.data.iloc[idx, 0]}.jpeg"
        image = Image.open(img_name).convert('RGB')
        ancillary_data = self.data.iloc[idx, 1:164].values.astype(np.float32)
        if self.transform:
            image = self.transform(image)
        return image, ancillary_data

def get_transforms():
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda img: random_color_jitter(img, probability=0.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform_train, transform_test

def extract_features(loader, model, device):
    model.eval()
    features_list = []
    ancillary_list = []
    with torch.no_grad():
        for images, ancillary_data in tqdm(loader):
            images = images.to(device)
            features = model(images).cpu().numpy()
            features_list.append(features)
            ancillary_list.append(ancillary_data)
    return np.vstack(features_list), np.vstack(ancillary_list)

def combine_features_and_ancillary(image_features, ancillary_data):
    return np.hstack([image_features, ancillary_data])

def init_XGBoost_tuning():
    # Hyperparameter spaces
    param_grid = {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [1000, 1500],
        'max_depth': [6, 8],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    search_spaces = {
        'learning_rate': (0.01, 0.1, 'uniform'),
        'n_estimators': (500, 2000),
        'max_depth': (3, 9),
        'subsample': (0.6, 1.0, 'uniform'),
        'colsample_bytree': (0.6, 1.0, 'uniform'),
        'min_child_weight': (1, 5),
        'gamma': (0, 0.2)
    }
    # Boosting and hyperparamter tuning models
    xgb_model_base = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42
    )
    xgb_model_manual = xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.01,
        n_estimators=1000,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    grid_search = GridSearchCV(
        estimator=xgb_model_base,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    bayes_search = BayesSearchCV(
        estimator=xgb_model_base,
        search_spaces=search_spaces,
        n_iter=10,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    return xgb_model_manual, grid_search, bayes_search

def train_and_tune_xgb_models(train_features, train_targets, tuning_flag):
    xgb_models = {}
    # Train an XGBoost model for each trait with hyperparameter tuning
    for i, trait in enumerate(["X4", "X11", "X18", "X26", "X50", "X3112"]):
        print(f'Running XGBoost with tuning on trait {trait}')
        # Initialize new models at each iteration, one of which will be used (based on the tuning flag)
        xgb_model_manual, grid_search, bayes_search = init_XGBoost_tuning()
        if tuning_flag == 'G':
            grid_search.fit(train_features, train_targets[:, i])
            best_model = grid_search.best_estimator_
            xgb_models[trait] = best_model
            best_params = grid_search.best_params_
        elif tuning_flag == 'B':
            print(f"Starting hyperparameter search at {get_time()}")
            bayes_search.fit(train_features, train_targets[:, i])
            print(f"Completed hyperparameter search at {get_time()}")
            best_model = bayes_search.best_estimator_
            xgb_models[trait] = best_model
            best_params = bayes_search.best_params_
        else:
            xgb_model_manual.fit(train_features, train_targets[:, i])
            best_model = xgb_model_manual
            xgb_models[trait] = best_model
            best_params = 'N/A'
        # Predict on training data and calculate R2 score
        train_predictions = best_model.predict(train_features)
        r2 = r2_score(train_targets[:, i], train_predictions)
        print(f'Best Parameters for {trait}: {best_params}')
        print(f'R2 score: {r2}')
    return xgb_models

def unnormalize_targets(df, target_param1, target_param2, min_max_norm=True):
    if min_max_norm:
        for i, column in enumerate(df.columns):
            df[column] = df[column] * (target_param2.iloc[i] - target_param1.iloc[i]) + target_param1.iloc[i]
    else:
        for i, column in enumerate(df.columns):
            df[column] = (df[column] * target_param2.iloc[i]) + target_param1.iloc[i]
    df = np.expm1(df)
    return df

# Create the final submission DataFrame
def save_predictions(predictions, test_csv, filename='20949894_Patel.csv'):
    test_data = pd.read_csv(test_csv)
    ids = test_data.iloc[:, 0]
    submission_df = pd.DataFrame(predictions, columns=["X4", "X11", "X18", "X26", "X50", "X3112"])
    submission_df = unnormalize_targets(submission_df, norm_param1[-6:], norm_param2[-6:])
    submission_df.insert(0, 'id', ids)
    submission_df.to_csv(filename, index=False)
    print(f'Predictions saved to {filename}')
    return filename

def send_email(subject, body, to_email, from_email, password, attachment_file):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_email, password)
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    with open(attachment_file, 'rb') as file:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(file.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename= {attachment_file}')
        msg.attach(part)
    server.send_message(msg)
    server.quit()

def get_time():
    current_time = datetime.now()
    formatted_time = current_time.strftime("%I:%M %p")
    return formatted_time

def prediction_pipline():
    tuning_flag = get_tuning_flag()

    # Define Data Transformations
    transform_train, transform_test = get_transforms()

    # Preprocess Data
    train_data = preprocess_data('data/train.csv', is_train=True)
    test_data = preprocess_data('data/test.csv', is_train=False)

    # Prepare DataLoaders
    train_dataset = PlantDataset('data/train_images', train_data, transform=transform_train)
    test_dataset = PlantDataset('data/test_images', test_data, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Extract Features
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = resnet.to(device)
    train_image_features, train_ancillary = extract_features(train_loader, resnet, device)
    test_image_features, test_ancillary = extract_features(test_loader, resnet, device)

    # Combine Features
    train_combined_features = combine_features_and_ancillary(train_image_features, train_ancillary)
    test_combined_features = combine_features_and_ancillary(test_image_features, test_ancillary)

    # Train and Tune Desired XGBoost Model for Each Target Trait (After Initialization)
    train_targets = train_data.iloc[:, -6:].values
    xgb_models = train_and_tune_xgb_models(train_combined_features, train_targets, tuning_flag)
    
    # Generate Predictions
    predictions = np.column_stack([model.predict(test_combined_features) for model in xgb_models.values()])
    
    # Save Results
    filename = save_predictions(predictions, 'data/test.csv')
    
    # Notify Results (optional)
    # send_email(
    #     subject='Your CSV File',
    #     body='Here is the CSV file you requested.',
    #     to_email='enter to_email here',
    #     from_email='enter from_email here',
    #     password='enter password here',
    #     attachment_file=filename
    # )

if __name__ == "__main__":
    prediction_pipline()
