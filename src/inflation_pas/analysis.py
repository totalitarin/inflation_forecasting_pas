"""Вложенная временная валидация линейной регрессии и Ridge."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_model(x, y, name):
    estimator = LinearRegression() if name == 'linear' else Ridge()
    pipeline = Pipeline([('scale', StandardScaler()), ('model', estimator)])
    grid = [{}] if name == 'linear' else {'model__alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]}
    search = GridSearchCV(pipeline, grid, cv=TimeSeriesSplit(n_splits=3),
                          scoring='neg_mean_absolute_error', error_score='raise')
    search.fit(x, y)
    return search


def interval(errors, prediction, coverage):
    """Эмпирический прогнозный интервал; минимум 12 прошлых ошибок."""
    if len(errors) < 12:
        return np.nan, np.nan
    tail = (1 - coverage) / 2
    low, high = np.quantile(errors, [tail, 1 - tail])
    # Включаем точечный прогноз, даже если прошлые ошибки одного знака.
    return prediction + min(low, 0), prediction + max(high, 0)


def evaluate(x, y, cpi, test_months=36):
    if test_months < 12 or len(x) - test_months < 36:
        raise ValueError('Нужно >=12 проверочных и >=36 обучающих строк после формирования лагов')
    rows = []
    errors = {name: [] for name in ('seasonal', 'linear', 'ridge')}
    for position in range(len(x) - test_months, len(x)):
        date = x.index[position]
        train_x, train_y = x.iloc[:position], y.iloc[:position]
        known = cpi.loc[cpi.index < date]
        scale = np.mean(np.abs(known.iloc[12:].to_numpy() - known.iloc[:-12].to_numpy()))
        models = {name: fit_model(train_x, train_y, name) for name in ('linear', 'ridge')}
        predictions = {name: float(model.predict(x.iloc[[position]])[0]) for name, model in models.items()}
        predictions['seasonal'] = float(x.iloc[position]['cpi_lag_12'])
        for name, prediction in predictions.items():
            row = {'date': date, 'model': name, 'actual': float(y.iloc[position]),
                   'prediction': prediction, 'mase_scale': scale}
            for level in (80, 95):
                row[f'lower_{level}'], row[f'upper_{level}'] = interval(errors[name], prediction, level / 100)
            rows.append(row)
            errors[name].append(row['actual'] - prediction)
    predictions = pd.DataFrame(rows)
    summaries = []
    for name, group in predictions.groupby('model'):
        absolute = (group.actual - group.prediction).abs()
        scales = group.mase_scale.replace(0, np.nan)
        result = {'model': name, 'MAE': absolute.mean(), 'RMSE': np.sqrt((absolute ** 2).mean()),
                  'MASE': (absolute / scales).mean(), 'n_test': len(group)}
        for level in (80, 95):
            valid = group.dropna(subset=[f'lower_{level}', f'upper_{level}'])
            result[f'coverage_{level}'] = (valid.actual.between(valid[f'lower_{level}'], valid[f'upper_{level}']).mean()
                                               if len(valid) else np.nan)
            result[f'n_interval_{level}'] = len(valid)
        summaries.append(result)
    return predictions, pd.DataFrame(summaries)


def forecast(x, y, future_x, predictions):
    forecasts, coefficients = [], []
    # Обе модели сохраняются; выбор не делается по тестовому периоду.
    for name in ('linear', 'ridge'):
        fitted = fit_model(x, y, name)
        value = float(fitted.predict(future_x)[0])
        past = predictions[predictions.model == name]
        errors = (past.actual - past.prediction).to_numpy()
        row = {'date': future_x.index[0].strftime('%Y-%m-%d'), 'model': name,
               'prediction': value, 'alpha': fitted.best_params_.get('model__alpha'),
               'inner_cv_MAE': -fitted.best_score_}
        for level in (80, 95):
            row[f'lower_{level}'], row[f'upper_{level}'] = interval(errors, value, level / 100)
        forecasts.append(row)
        model = fitted.best_estimator_.named_steps['model']
        coefficients.extend({'model': name, 'feature': col, 'coefficient_standardized': float(coef)}
                            for col, coef in zip(x.columns, model.coef_))
    return forecasts, pd.DataFrame(coefficients)
