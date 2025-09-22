import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
import joblib 

data = pd.read_csv('last_satis.csv')

if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis=1)

y = data['Ümumi satış']
X = data.drop(['Ümumi satış', 'Tarix', 'Transaksiya_id'], axis=1)
print("The data is prepared. Signs used:", X.columns.tolist())

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
        ('cat', categorical_transformer, make_column_selector(dtype_include='object'))
    ]
)

pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

pipeline_rf.fit(X, y) 

model_filename = 'regressor_pipeline.joblib'
joblib.dump(pipeline_rf, model_filename)
print(f"The model has been successfully saved to a file: {model_filename}")