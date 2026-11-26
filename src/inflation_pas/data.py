"""Загрузка, проверка и формирование признаков без будущих наблюдений."""
from io import StringIO
from pathlib import Path
import json
import time
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

MONTHS = 'январь февраль март апрель май июнь июль август сентябрь октябрь ноябрь декабрь'.split()


def download(url, path, log_path):
    """Сохраняет ответ и журнал; повторяет временно неуспешные запросы."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {'url': url, 'downloaded_at': pd.Timestamp.now(tz='UTC').isoformat()}
    try:
        for attempt in range(3):
            try:
                with urlopen(Request(url, headers={'User-Agent': 'InflationPAS/1.0'}), timeout=30) as response:
                    body = response.read()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        path.write_bytes(body)
        event.update(status='ok', bytes=len(body), path=str(path))
        return body
    except Exception as exc:
        event.update(status='error', error=str(exc))
        raise
    finally:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')


def parse_cpi_html(text):
    """Таблица Audit-it: ИПЦ к предыдущему месяцу, 100,xx -> 0,xx %."""
    records = []
    for table in pd.read_html(StringIO(text), decimal=',', thousands=None):
        for column in table.columns[1:]:
            try:
                year = int(str(column))
            except ValueError:
                continue
            for _, row in table.iterrows():
                month = str(row.iloc[0]).strip().lower()
                if month not in MONTHS:
                    continue
                value = pd.to_numeric(str(row[column]).replace(',', '.'), errors='coerce')
                if pd.notna(value):
                    records.append({'date': f'{year}-{MONTHS.index(month)+1:02d}-01', 'cpi_mom': value - 100})
    if not records:
        raise ValueError('Таблица месячного ИПЦ не распознана')
    return pd.DataFrame(records)


def collect(start_year, end_year, raw_dir):
    """ИПЦ из вторичной публикации Росстата и USD/RUB из XML Банка России."""
    raw_dir = Path(raw_dir)
    frames = []
    # На каждой странице пять лет; первая страница содержит текущий год.
    current = pd.Timestamp.now().year
    for interval in range(1, (current - start_year) // 5 + 2):
        url = 'https://www.audit-it.ru/inform/inflation/'
        if interval > 1:
            url += f'?years_interval={interval}'
        body = download(url, raw_dir / f'cpi_{interval}.html', raw_dir / 'downloads.jsonl')
        frames.append(parse_cpi_html(body.decode('utf-8')))
    cpi = pd.concat(frames).drop_duplicates()
    cpi['date'] = pd.to_datetime(cpi['date'])
    cpi = cpi[cpi.date.dt.year.between(start_year, end_year)]
    rates = []
    for year in range(start_year - 1, end_year + 1):
        url = ('https://www.cbr.ru/scripts/XML_dynamic.asp?'
               f'date_req1=01/01/{year}&date_req2=31/12/{year}&VAL_NM_RQ=R01235')
        body = download(url, raw_dir / f'usd_{year}.xml', raw_dir / 'downloads.jsonl')
        for record in ET.fromstring(body).findall('Record'):
            value = float(record.findtext('Value').replace(',', '.'))
            nominal = float(record.findtext('Nominal').replace(',', '.'))
            rates.append((pd.to_datetime(record.attrib['Date'], dayfirst=True), value / nominal))
    if not rates:
        raise ValueError('Банк России вернул пустой ряд USD/RUB')
    usd = pd.DataFrame(rates, columns=['date', 'usd_rub']).drop_duplicates('date').set_index('date')
    usd = usd.resample('MS').mean()
    usd['usd_rub_mom'] = usd.usd_rub.pct_change(fill_method=None) * 100
    return validate(cpi.merge(usd[['usd_rub_mom']], on='date', how='left'))


def validate(frame):
    frame = frame.copy()
    required = {'date', 'cpi_mom'}
    if not required.issubset(frame.columns):
        raise ValueError('Нужны столбцы date и cpi_mom (прирост в %, не индекс 100+)')
    dates = pd.to_datetime(frame.pop('date'), errors='raise')
    if not dates.dt.is_month_start.all():
        raise ValueError('date должен обозначать первое число месяца')
    frame.index = pd.DatetimeIndex(dates, name='date')
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError('Повторяющиеся месяцы')
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq='MS')
    if not frame.index.equals(expected.rename('date')):
        raise ValueError('В месячном ряду есть пропуски')
    for col in frame:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Пропущенные или бесконечные значения: исправьте исходную выгрузку')
    if not frame.cpi_mom.between(-20, 50).all():
        raise ValueError('Неправдоподобный cpi_mom: ожидается прирост, например 0.8, а не 100.8')
    if len(frame) < 72:
        raise ValueError('Нужно не менее 72 непрерывных месячных наблюдений')
    return frame


def load_csv(path):
    return validate(pd.read_csv(path))


def make_features(frame):
    """Индекс строки — прогнозируемый месяц; все наблюдаемые признаки сдвинуты."""
    index = frame.index.append(pd.DatetimeIndex([frame.index[-1] + pd.offsets.MonthBegin()]))
    history = frame.reindex(index)
    x = pd.DataFrame(index=index)
    for lag in (1, 2, 3, 12):
        x[f'cpi_lag_{lag}'] = history.cpi_mom.shift(lag)
    for window in (3, 12):
        x[f'cpi_mean_{window}'] = history.cpi_mom.shift(1).rolling(window).mean()
    # Только явно поддержанные факторы: неизвестные колонки не становятся признаками.
    if 'usd_rub_mom' in history:
        for lag in (1, 2, 3):
            x[f'usd_change_lag_{lag}'] = history.usd_rub_mom.shift(lag)
    x['month_sin'] = np.sin(2 * np.pi * index.month / 12)
    x['month_cos'] = np.cos(2 * np.pi * index.month / 12)
    ready = x.dropna()
    train = ready.index.intersection(frame.index)
    return ready.loc[train], frame.loc[train, 'cpi_mom'], ready.iloc[[-1]]
