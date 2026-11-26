"""Основной файл ПР3: импортирует и вызывает функции независимых модулей."""
import argparse
from datetime import date
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from inflation_pas.data import collect, load_csv, make_features
from inflation_pas.analysis import evaluate, forecast
from inflation_pas.visualization import save_figures


def main():
    parser = argparse.ArgumentParser(description='Прогноз месячной инфляции: LinearRegression и Ridge')
    parser.add_argument('--input', type=Path, help='CSV: date,cpi_mom[,usd_rub_mom]')
    parser.add_argument('--download', action='store_true', help='Загрузить ИПЦ Audit-it и курс Банка России')
    parser.add_argument('--start-year', type=int, default=2010)
    parser.add_argument('--end-year', type=int, default=date.today().year)
    parser.add_argument('--test-months', type=int, default=36)
    parser.add_argument('--output', type=Path, default=Path('reports/generated/pr3'))
    args = parser.parse_args()
    if bool(args.input) == args.download:
        parser.error('Укажите ровно один вариант: --input или --download')
    history = (collect(args.start_year, args.end_year, Path('data/raw'))
               if args.download else load_csv(args.input))
    x, y, future_x = make_features(history)
    predictions, metrics = evaluate(x, y, history.cpi_mom, args.test_months)
    forecasts, coefficients = forecast(x, y, future_x, predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.output / 'monthly.csv')
    x.assign(target=y).to_csv(args.output / 'features.csv')
    predictions.to_csv(args.output / 'backtest.csv', index=False)
    metrics.to_csv(args.output / 'metrics.csv', index=False)
    coefficients.to_csv(args.output / 'coefficients.csv', index=False)
    (args.output / 'forecast.json').write_text(json.dumps(forecasts, ensure_ascii=False, indent=2), encoding='utf-8')
    save_figures(history, predictions, metrics, coefficients, forecasts, args.output)
    (args.output / 'index.html').write_text('''<!doctype html><html lang="ru"><meta charset="utf-8">
<title>Практическая работа 3</title><h1>Прогноз месячной инфляции</h1>
<p>Последовательно проверялись сезонный прогноз, линейная регрессия и Ridge.</p>
<ul><li><a href="forecast.html">История и прогноз</a></li><li><a href="metrics.html">Метрики</a></li>
<li><a href="coefficients.html">Коэффициенты</a></li><li><a href="metrics.csv">Таблица метрик</a></li>
<li><a href="forecast.json">Прогноз и интервалы</a></li></ul></html>''', encoding='utf-8')
    print(metrics.to_string(index=False))
    print(json.dumps(forecasts, ensure_ascii=False, indent=2))
    print(f'Результаты: {args.output.resolve() / "index.html"}')


if __name__ == '__main__':
    main()
