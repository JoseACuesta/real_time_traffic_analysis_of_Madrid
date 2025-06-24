import polars as pl
import numpy as np
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

from scipy import sparse

from transform_data import raw_data_from_polars_dataframe, transform_polars_dataframe

def split_train_and_test_data(df: pl.DataFrame):
    
    is_2024 = df['year'] == 2024.0

    X = df.drop('carga')
    y = df['carga']

    X_train = X.filter(~is_2024)
    X_test = X.filter(is_2024)

    y_train = y.filter(~is_2024)
    y_test = y.filter(is_2024)

    return X_train, X_test, y_train, y_test

def normalize_and_scale_data_polars(X_train: pl.DataFrame, X_test: pl.DataFrame) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    
    categorical_column = ["tipo_elem"]
    numerical_columns = [col for col in X_train.columns if col not in categorical_column]

    
    X_train_cat = X_train.select(categorical_column).to_pandas()
    X_test_cat = X_test.select(categorical_column).to_pandas()

    X_train_num = X_train.select(numerical_columns).to_numpy()
    X_test_num = X_test.select(numerical_columns).to_numpy()

    
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    se = StandardScaler()

    
    X_train_cat_ohe = ohe.fit_transform(X_train_cat)
    X_test_cat_ohe = ohe.transform(X_test_cat)

    X_train_num_scaled = se.fit_transform(X_train_num)
    X_test_num_scaled = se.transform(X_test_num)

    
    X_train_final = sparse.hstack([X_train_num_scaled, X_train_cat_ohe], format='csr')
    X_test_final = sparse.hstack([X_test_num_scaled, X_test_cat_ohe], format='csr')

    return X_train_final, X_test_final

def train_model(X_train_final: np.ndarray, y_train: pl.Series) -> RandomForestRegressor:

    y_train = y_train.to_numpy()

    rfr = RandomForestRegressor(n_jobs=-1)

    param_grid = {
        'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
        'max_depth': [10, 20],
        'min_samples_split': [10],
        'min_samples_leaf': [2],
    }

    grid = GridSearchCV(param_grid=param_grid, estimator=rfr, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(X_train_final, y_train)
    best_model = grid.best_estimator_
    
    return best_model

def test_model(best_model: RandomForestRegressor, X_test_final: np.ndarray, y_test: pl.Series) -> np.ndarray:
    
    y_test = y_test.to_numpy()

    y_pred = best_model.predict(X_test_final)
    return y_pred

def main():
    df = raw_data_from_polars_dataframe(path=Path('../../../data-preprocessing/src/data_preprocessing/data/final_data.csv'))
    df_transformed = transform_polars_dataframe(df)
    X_train, X_test, y_train, y_test = split_train_and_test_data(df_transformed)
    X_train_final, X_test_final = normalize_and_scale_data_polars(X_train, X_test)
    best_model = train_model(X_train_final, y_train)
    y_pred = test_model(best_model, X_test_final, y_test)
    print("Random Forest Regressor best: ", str(np.round(mean_absolute_error(y_test, y_pred), 2)))

if __name__ == "__main__":
    main()