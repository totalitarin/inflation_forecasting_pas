"""Функции визуализации не обучают модели и не изменяют данные."""
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go


def save_figures(history, predictions, metrics, coefficients, forecasts, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    figure = go.Figure()
    figure.add_scatter(x=history.index, y=history.cpi_mom, name='Факт', line_color='#222222')
    colors = {'linear': '#3678b5', 'ridge': '#cc6622', 'seasonal': '#777777'}
    for name, group in predictions.groupby('model'):
        figure.add_scatter(x=group.date, y=group.prediction, name=name, line_color=colors[name])
    for row in forecasts:
        figure.add_scatter(x=[row['date']], y=[row['prediction']], name=f"Прогноз {row['model']}",
                          mode='markers', marker={'size': 10, 'color': colors[row['model']]},
                          error_y={'type': 'data', 'symmetric': False,
                                   'array': [row['upper_95'] - row['prediction']],
                                   'arrayminus': [row['prediction'] - row['lower_95']]})
    figure.update_layout(title='Месячная инфляция и прогноз на один месяц',
                         xaxis_title='Месяц', yaxis_title='Инфляция, % м/м', template='plotly_white')
    figure.write_html(directory / 'forecast.html', include_plotlyjs=True)
    long = metrics.melt(id_vars='model', value_vars=['MAE', 'RMSE'], var_name='metric', value_name='error')
    px.bar(long, x='model', y='error', color='metric', barmode='group',
           title='Ошибка прогноза в процентных пунктах', template='plotly_white').write_html(directory / 'metrics.html')
    px.bar(coefficients, x='coefficient_standardized', y='feature', color='model', barmode='group',
           title='Коэффициенты стандартизованных признаков — не причинные эффекты',
           template='plotly_white').write_html(directory / 'coefficients.html')
    return ['forecast.html', 'metrics.html', 'coefficients.html']
