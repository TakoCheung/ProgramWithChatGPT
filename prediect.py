import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from datetime import date
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    df['date'] = pd.to_datetime(df['Draw Date'])
    
    # Convert the number strings into individual columns
    df[['n0', 'n1', 'n2', 'n3', 'n4', 'n5']] = df['Winning Numbers'].str.split(' ', expand=True).astype(int)
    df.drop([ 'Draw Date', 'Winning Numbers', 'Multiplier'], axis=1, inplace=True)

    df = df.melt(id_vars=['date','n5'], var_name='number_id', value_name='number')
    df.drop('number_id', axis=1, inplace=True)

    return df

def train_and_evaluate_model(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

def predict_next_numbers(pipeline, last_date):
    next_date = last_date + pd.Timedelta(days=3)
    next_date_features = pd.DataFrame(
        [next_date], 
        columns=['date']
    )
    next_numbers = pipeline.predict(next_date_features)
    return np.round(next_numbers).astype(int)


data = pd.read_csv('powerball_winning_numbers.csv')
data = preprocess_data(data)
#print(data)
X = data.drop(['number', 'n5'], axis=1)
y = data[['number', 'n5']]

y_number = data['number']
y_n5 = data['n5']

X_train_number, X_test_number, y_train_number, y_test_number = train_test_split(X, y_number, test_size=0.2, random_state=42)
X_train_n5, X_test_n5, y_train_n5, y_test_n5 = train_test_split(X, y_n5, test_size=0.2, random_state=42)

pipelines = [
    Pipeline([('scaler', StandardScaler()), ('linear_regression', LinearRegression())]),
    Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))]),
    Pipeline([('scaler', StandardScaler()), ('lasso', Lasso(alpha=1.0))]),
    Pipeline([('scaler', StandardScaler()), ('knr', KNeighborsRegressor())]),
    Pipeline([('scaler', StandardScaler()), ('rfr', RandomForestRegressor(n_estimators=100, random_state=42))])
]

last_date = data['date'].iloc[-1]
for i, pipeline in enumerate(pipelines):
    r2_number = train_and_evaluate_model(pipeline, X_train_number, X_test_number, y_train_number, y_test_number)
    next_numbers = predict_next_numbers(pipeline, last_date)
    print(f"Next numbers predicted by {pipeline.steps[1]} with R2-score: {r2_number}: {next_numbers}")
    
print("-------------------------------------------")

for i, pipeline in enumerate(pipelines):
    r2_n5 = train_and_evaluate_model(pipeline, X_train_n5, X_test_n5, y_train_n5, y_test_n5)
    next_numbers = predict_next_numbers(pipeline, last_date)
    print(f"Next n5 predicted by {pipeline.steps[1]} with R2-score: {r2_number}: {next_numbers}")