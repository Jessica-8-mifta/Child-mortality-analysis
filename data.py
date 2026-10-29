"""World Bank ingestion and explicit complete-case validation."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlencode
import time
import numpy as np
import pandas as pd

CODES = {'u5_mort': 'SH.DYN.MORT', 'measles_imm': 'SH.IMM.MEAS',
         'gdp_pc': 'NY.GDP.PCAP.CD', 'water': 'SH.H2O.BASW.ZS', 'sanit': 'SH.STA.BASS.ZS'}
FEATURES = ['measles_imm', 'gdp_pc', 'water', 'sanit']

def fetch_pages(path, year=None):
    rows, payloads, page = [], [], 1
    while True:
        params = {'format': 'json', 'per_page': 20000, 'page': page}
        if year is not None:
            params['date'] = year
        url = 'https://api.worldbank.org/v2/' + path + '?' + urlencode(params)
        for attempt in range(3):
            try:
                with urlopen(url, timeout=60) as response:
                    payload = json.load(response)
                if not isinstance(payload, list) or len(payload) != 2 or not isinstance(payload[1], list):
                    raise ValueError('Unexpected World Bank response: ' + url)
                break
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        payloads.append({'url': url, 'response': payload})
        rows.extend(payload[1])
        if page >= int(payload[0]['pages']):
            return rows, payloads
        page += 1

def download(year, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    countries, responses = fetch_pages('country')
    # Includes economies/territories classified as non-aggregate by the provider.
    eligible = {c['id']: c['name'] for c in countries if c['region']['id'] != 'NA'}
    frame = pd.DataFrame(index=pd.Index(sorted(eligible), name='iso3'))
    frame['country'] = pd.Series(eligible)
    frame['year'] = year
    for name, code in CODES.items():
        rows, pages = fetch_pages('country/all/indicator/' + code, year)
        responses.extend(pages)
        values = {r['countryiso3code']: r['value'] for r in rows
                  if r['countryiso3code'] in eligible and str(r['date']) == str(year)}
        frame[name] = pd.Series(values, dtype=float)
    path = directory / f'world_bank_{year}.csv'
    frame.reset_index().to_csv(path, index=False)
    (directory / f'api_responses_{year}.json').write_text(json.dumps(responses))
    metadata = {'retrieved_at_utc': datetime.now(timezone.utc).isoformat(), 'year': year,
                'indicators': CODES, 'source': 'World Bank Indicators API v2',
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (directory / f'metadata_{year}.json').write_text(json.dumps(metadata, indent=2))
    return path

def validate(frame, year):
    required = ['iso3', 'year', *CODES]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    frame = frame.copy()
    if frame['iso3'].isna().any() or not frame['iso3'].astype(str).str.fullmatch(r'[A-Z0-9]{3}').all():
        raise ValueError('iso3 must contain three-character economy codes')
    if frame['iso3'].duplicated().any():
        raise ValueError('Duplicate economy codes')
    if not pd.to_numeric(frame['year'], errors='raise').eq(year).all():
        raise ValueError('Input contains a different year')
    for col in CODES:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
    values = frame[list(CODES)]
    invalid = pd.Series(False, index=frame.index)
    invalid |= ((values[['u5_mort', 'gdp_pc']] <= 0) | np.isinf(values[['u5_mort', 'gdp_pc']])).any(axis=1)
    for col in ['measles_imm', 'water', 'sanit']:
        invalid |= frame[col].notna() & ~frame[col].between(0, 100)
    incomplete = values.isna().any(axis=1)
    audit = frame[['iso3']].copy()
    audit['missing_values'] = incomplete
    audit['invalid_values'] = invalid
    audit['included'] = ~(incomplete | invalid)
    clean = frame.loc[audit['included']].set_index('iso3').sort_index()
    if len(clean) < 25:
        raise ValueError('At least 25 valid complete economies are required')
    return clean, audit, values.isna().mean().rename('missing_share')

def features(frame):
    result = frame[FEATURES].astype(float).copy()
    result['gdp_pc'] = np.log(result['gdp_pc'])
    return result
