"""
Thesis regression pipeline.
Estimates announcement-surprise reactions under fiscal policy uncertainty
and government shutdowns for ES and ZN futures.
Output: results/regression_results.xlsx
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os
from pathlib import Path

# Event-time helpers: announcements sharing a date+minute share one return

def _norm_time(t):
    if pd.isna(t):
        return None
    if hasattr(t, 'hour'):
        return f'{t.hour:02d}:{t.minute:02d}'
    parts = str(t).split(':')
    return f'{int(parts[0]):02d}:{int(parts[1]):02d}'

def add_event_id(df):
    df = df.copy()
    df['event_time'] = df['release_time'].apply(_norm_time)
    df['event_id'] = (df['release_date'].dt.strftime('%Y-%m-%d')
                      + '_' + df['event_time'].astype(str))
    return df

def verify_shared_returns(df, ret_cols, label=''):
    """Within each event_id, the return columns must be identical (one window).
    Any mismatch signals a data problem."""
    for col in ret_cols:
        if col not in df.columns:
            continue
        bad = 0
        sub = df.dropna(subset=[col])
        for _, grp in sub.groupby('event_id'):
            if len(grp) > 1 and grp[col].round(6).nunique() > 1:
                bad += 1
        flag = 'OK' if bad == 0 else f'*** {bad} MISMATCHES ***'
        print(f"    [{label}] shared-return check {col}: {flag}")

def dedup_returns(df, ret_cols):
    """Assign one representative return per event_id to all its rows (long kept)."""
    df = df.copy()
    for col in ret_cols:
        if col not in df.columns:
            continue
        df[col] = df.groupby('event_id')[col].transform(
            lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    return df

def type_token(ty):
    return (str(ty).replace(' ', '_').replace('-', '_').replace('&', 'and')
              .replace('(', '').replace(')', '').replace('/', '_').replace('.', ''))

def make_wide(df, surprise_col='surprise_std', ret_cols=('ret_sym5',),
              carry_cols=(), period_col='period'):
    """Reshape de-duplicated long -> event-time WIDE.
    One row per event_id; S_<token> = that type's surprise (0.0 if absent),
    has_<token> = 1 if the type was present (preserves genuine zero surprises).
    Same-type collisions keep the LATEST reference period. carry_cols (returns,
    regime flags, FPU) are date-keyed and taken once per event_id."""
    df = df.copy()
    if period_col in df.columns:
        df['_pdt'] = pd.to_datetime(df[period_col], dayfirst=True, errors='coerce')
    else:
        df['_pdt'] = pd.NaT

    types = sorted(df['type'].dropna().unique())
    token_map = {ty: type_token(ty) for ty in types}

    carry = [c for c in (list(ret_cols) + list(carry_cols)) if c in df.columns]
    base = (df[['event_id', 'release_date', 'event_time'] + carry]
            .groupby('event_id', as_index=False).first())

    for ty, tok in token_map.items():
        s_col, h_col = f'S_{tok}', f'has_{tok}'
        sub = df[df['type'] == ty].sort_values('_pdt', na_position='first')
        picked = (sub.groupby('event_id')
                     .agg(**{s_col: (surprise_col, 'last')}).reset_index())
        base = base.merge(picked, on='event_id', how='left')
        base[h_col] = base[s_col].notna().astype(int)
        base[s_col] = base[s_col].fillna(0.0)
    return base, token_map

def build_wide_formula(dep, token_map, regime=None, present_tokens=None):
    """Construct the joint type-specific formula on wide columns.
      dep ~ [regime] + has_<t> (type FE) + S_<t> (slopes)
            + S_<t>:regime + S_<t> (interaction with regime if given)
    If regime is None -> FPU-style handled by caller passing regime='FPU_std'
    via the interaction term. present_tokens limits to types actually present
    (avoids all-zero columns from types that never appear in this subsample)."""
    toks = present_tokens if present_tokens is not None else list(token_map.values())
    has_terms = [f'has_{t}' for t in toks]
    s_terms = [f'S_{t}' for t in toks]
    terms = []
    if regime is not None:
        terms.append(regime)
    terms += has_terms[1:]          # type fixed effects (drop one for identification)
    terms += s_terms                # per-type surprise slopes
    if regime is not None:
        terms += [f'{s}:{regime}' for s in s_terms]   # per-type regime interactions
    return f'{dep} ~ ' + ' + '.join(terms)

RUN_DESCRIPTIVES    = True
RUN_EPU_REGRESSIONS = True
RUN_SHUTDOWN_MODELS = True
RUN_PER_TYPE        = True
RUN_ASYMMETRY       = True
RUN_ROBUSTNESS      = True
RUN_SAVE            = True
RUN_OVERLAP_CHECK   = True
RUN_DESCRIPTIVE_TABLES  = True

# FILE PATHS
# Announcement mastersheets are committed under ./data and resolve automatically.
# The ES/ZN 5-min bars are licensed and not distributed — see README, "Data
# availability". Place them in ./data/futures, or point FUTURES_DIR elsewhere:
#     FUTURES_DIR=/path/to/bars python fpu_event_study.py
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / 'data'
FUTURES_DIR = Path(os.environ.get('FUTURES_DIR', DATA_DIR / 'futures'))
OUTPUT_DIR  = BASE_DIR / 'results'
OUTPUT_DIR.mkdir(exist_ok=True)

PATH_ES       = FUTURES_DIR / 'ES_5min_full(2010-2026).xlsx'
PATH_ZN       = FUTURES_DIR / 'ZN_5min_full(2010-2026)_v2.csv'
PATH_EPU      = DATA_DIR / 'mastersheet_epu.xlsx'
PATH_SHUTDOWN = DATA_DIR / 'mastersheet_shutdowns.xlsx'

# 1. LOAD DATA
print("=" * 60)
print("SECTION 1: LOADING DATA")
print("=" * 60)

# ES futures (Excel) — required; every dependent variable is built from these bars
if not PATH_ES.exists():
    raise SystemExit(
        f"ES 5-min bars not found at {PATH_ES}\n"
        "These are licensed data and are not distributed with this repository.\n"
        "See README, 'Data availability', for the required columns and layout.")
es = pd.read_excel(PATH_ES)
es.columns = es.columns.str.strip().str.lower().str.replace(' ', '_')
if 'ts_event' in es.columns:
    es.rename(columns={'ts_event': 'datetime'}, inplace=True)
es['datetime'] = pd.to_datetime(es['datetime'])
es = es.sort_values('datetime').reset_index(drop=True)
print(f"  ES futures: {len(es):,} bars from {es['datetime'].min()} to {es['datetime'].max()}")

# ZN futures
try:
    zn = pd.read_csv(PATH_ZN)
    zn.columns = zn.columns.str.strip().str.lower().str.replace(' ', '_')
    if 'ts_event' in zn.columns:
        zn.rename(columns={'ts_event': 'datetime'}, inplace=True)
    zn['datetime'] = (pd.to_datetime(zn['datetime'], utc=True)
                        .dt.tz_convert('America/New_York')
                        .dt.tz_localize(None))
    zn = zn.sort_values('datetime').reset_index(drop=True)
    HAS_ZN = True
    print(f"  ZN futures: {len(zn):,} bars from {zn['datetime'].min()} to {zn['datetime'].max()}")
except Exception as e:
    print(f"  ZN futures: NOT LOADED ({e})")
    HAS_ZN = False
    zn = None

# EPU mastersheet — load FPU_raw alongside other columns
epu_raw = pd.read_excel(PATH_EPU, sheet_name='Sheet1')
epu_raw.columns = epu_raw.columns.str.strip()
epu_raw = epu_raw.rename(columns={
    'Release date': 'release_date', 'Release time': 'release_time',
    'Type': 'type', 'Surprise_standardized': 'surprise_std',
    'In Fiscal stress?': 'fiscal_stress'
})
for alt in ['Suprise_raw', 'Surprise_raw']:
    if alt in epu_raw.columns:
        epu_raw = epu_raw.rename(columns={alt: 'surprise_raw'})

fpu_col = None
for candidate in ['FPU_raw', 'FPU', 'Fiscal_Policy_EPU', 'EPU_fiscal', 'EPU_FP']:
    if candidate in epu_raw.columns:
        fpu_col = candidate
        break

keep_cols = ['release_date', 'release_time', 'type', 'surprise_std', 'fiscal_stress']
if 'surprise_raw' in epu_raw.columns:
    keep_cols.append('surprise_raw')
if 'Period' in epu_raw.columns:
    keep_cols.append('Period')
if fpu_col:
    keep_cols.append(fpu_col)
    HAS_FPU = True
    print(f"  FPU raw column found: '{fpu_col}'")
else:
    HAS_FPU = False
    print("  WARNING: No FPU_raw column — continuous FPU models skipped")

epu = epu_raw[keep_cols].copy()
if 'Period' in epu.columns:
    epu = epu.rename(columns={'Period': 'period'})
if fpu_col:
    epu = epu.rename(columns={fpu_col: 'FPU_raw'})
epu['release_date'] = pd.to_datetime(epu['release_date'], dayfirst=True)
epu['fiscal_stress'] = epu['fiscal_stress'].astype(float).fillna(0).astype(int)
epu['type'] = epu['type'].astype(str).str.strip()

# Shutdown mastersheet
shut_raw = pd.read_excel(PATH_SHUTDOWN, sheet_name='Sheet1')
shut_raw.columns = shut_raw.columns.str.strip()
shut_raw = shut_raw.rename(columns={
    'Release date': 'release_date', 'Release time': 'release_time',
    'Type': 'type', 'Surprise_standardized': 'surprise_std',
    'Shutdown': 'shutdown', 'Shutdown_id': 'shutdown_id',
    'Pre_Window': 'pre_window', 'Post_Window': 'post_window',
    'Dummy_delayed': 'delayed_release'
})
for alt in ['Suprise_raw', 'Surprise_raw']:
    if alt in shut_raw.columns:
        shut_raw = shut_raw.rename(columns={alt: 'surprise_raw'})

shut_keep = ['release_date', 'release_time', 'type', 'surprise_std',
             'shutdown', 'shutdown_id', 'pre_window', 'post_window',
             'delayed_release']
if 'surprise_raw' in shut_raw.columns:
    shut_keep.append('surprise_raw')
if 'Period' in shut_raw.columns:
    shut_keep.append('Period')
shut = shut_raw[shut_keep].copy()
if 'Period' in shut.columns:
    shut = shut.rename(columns={'Period': 'period'})
shut['release_date'] = pd.to_datetime(shut['release_date'], dayfirst=True)
for col in ['shutdown', 'pre_window', 'post_window']:
    shut[col] = (pd.to_numeric(shut[col], errors='coerce')
                 .fillna(0).round().astype(int))
shut['delayed_release'] = (pd.to_numeric(shut['delayed_release'], errors='coerce')
                           .fillna(0).round().astype(int))
shut['type'] = shut['type'].astype(str).str.strip()

print(f"  EPU mastersheet: {len(epu):,} events, {epu['type'].nunique()} types")
print(f"  Shutdown mastersheet: {len(shut):,} events, {shut['delayed_release'].sum()} delayed")

# 2. COMPUTE RETURNS IN BASIS POINTS
print("\n" + "=" * 60)
print("SECTION 2: COMPUTING RETURNS")
print("=" * 60)

def get_return_bps(release_date, release_time, prices_df, pre_min, post_min,
                   mode='close_to_close'):
    """Return in bps between two prices around the announcement.

    The announcement minute t0 is the START of the announcement bar (bars are
    timestamped at their start). Two modes:

    - mode='ann_bar' (MAIN window, [0,+5]): open of the announcement bar (= price
      at t0, the release instant) -> close of the announcement bar (= price at
      t0+5). This is the immediate five-minute post-announcement reaction
      (Andersen et al., 2003 convention). post_min is ignored.

    - mode='close_to_close' (longer windows): close of the bar at t0-pre_min ->
      close of the bar at t0+post_min-5 (so [0,+10] spans the announcement bar
      plus the next, i.e. t0 open .. t0+10 close). For pre_min=0 we anchor the
      pre-price at the announcement bar OPEN to stay consistent with the main
      window; otherwise we use the close of the bar pre_min minutes earlier.
    """
    if isinstance(release_time, str):
        parts = release_time.split(':')
        h, m = int(parts[0]), int(parts[1])
    else:
        h, m = release_time.hour, release_time.minute
    t0 = pd.Timestamp(release_date.year, release_date.month, release_date.day, h, m, 0)

    if mode == 'ann_bar':
        row = prices_df[prices_df['datetime'] == t0]
        if row.empty:
            return np.nan
        p_pre = row['open'].values[0]
        p_post = row['close'].values[0]
        if p_pre <= 0 or pd.isna(p_pre) or pd.isna(p_post):
            return np.nan
        return np.log(p_post / p_pre) * 10000

    # close_to_close (longer/robustness windows)
    if pre_min == 0:
        # anchor pre-price at the announcement bar OPEN (price at t0)
        row_pre = prices_df[prices_df['datetime'] == t0]
        p_pre = row_pre['open'].values[0] if not row_pre.empty else np.nan
    else:
        row_pre = prices_df[prices_df['datetime'] == (t0 - pd.Timedelta(minutes=pre_min))]
        p_pre = row_pre['close'].values[0] if not row_pre.empty else np.nan
    # post-price = close of the bar ending at t0+post_min, i.e. bar starting at t0+post_min-5
    t_post_bar = t0 + pd.Timedelta(minutes=post_min - 5)
    row_post = prices_df[prices_df['datetime'] == t_post_bar]
    p_post = row_post['close'].values[0] if not row_post.empty else np.nan
    if pd.isna(p_pre) or pd.isna(p_post) or p_pre <= 0:
        return np.nan
    return np.log(p_post / p_pre) * 10000

# MAIN window: ret_sym5 now holds the [0,+5] announcement-bar open->close return
WINDOWS = {'ret_sym5': (0, 5, 'ann_bar'),
           'ret_10min': (0, 10, 'close_to_close'),
           'ret_15min': (0, 15, 'close_to_close'),
           'ret_30min': (0, 30, 'close_to_close'),
           'ret_sym10': (5, 10, 'close_to_close')}

def compute_returns(df, prices_df, label='ES'):
    for col, (pre, post, mode) in WINDOWS.items():
        df[col] = df.apply(
            lambda r: get_return_bps(r['release_date'], r['release_time'],
                                     prices_df, pre, post, mode), axis=1)
        if col == 'ret_sym5':
            window_label = '[0,+5] MAIN (ann-bar open->close)'
        elif col == 'ret_sym10':
            window_label = '[-5,+5] symmetric 10-min (robustness)'
        else:
            window_label = f"[0,+{post}]"
        print(f"    {label} {col} {window_label}: {df[col].notna().sum()}/{len(df)} matched")
    return df

def compute_zn_returns(df, zn_df, label='ZN'):
    for col, (pre, post, mode) in WINDOWS.items():
        zn_col = col.replace('ret_', 'zn_')
        df[zn_col] = df.apply(
            lambda r: get_return_bps(r['release_date'], r['release_time'],
                                     zn_df, pre, post, mode), axis=1)
        if col == 'ret_sym5':
            window_label = '[0,+5] MAIN (ann-bar open->close)'
        elif col == 'ret_sym10':
            window_label = '[-5,+5] symmetric 10-min (robustness)'
        else:
            window_label = f"[0,+{post}]"
        print(f"    {label} {zn_col} {window_label}: {df[zn_col].notna().sum()}/{len(df)} matched")
    return df

print("  EPU sheet (ES):")
epu = compute_returns(epu, es, 'ES')
if HAS_ZN:
    print("  EPU sheet (ZN):")
    epu = compute_zn_returns(epu, zn, 'ZN')

print("  Shutdown sheet (ES):")
shut = compute_returns(shut, es, 'ES')
if HAS_ZN:
    print("  Shutdown sheet (ZN):")
    shut = compute_zn_returns(shut, zn, 'ZN')

# 2b. STANDARDIZE SURPRISES IN CODE (uniform, all 16 types)
# Supersedes the Excel 'Surprise_standardized' column, which had been computed on
# the full sample including COVID. Recomputing here keeps sigma identical across
# the EPU and shutdown frames; the full-sample variant is retained as
# surprise_std_covid for the Section 10c robustness check.
print("\n" + "=" * 60)
print("SECTION 2b: STANDARDIZING SURPRISES (uniform, non-COVID sigma)")
print("=" * 60)

def _covid_mask(df):
    return (df['release_date'] >= '2020-04-01') & (df['release_date'] <= '2020-07-31')

def standardize_surprises(df, sigma_map=None, label=''):
    """Compute surprise_std = surprise_raw / sigma_j per type. If sigma_map is
    given, use it (ensures identical sigma across frames); otherwise compute the
    non-COVID sample SD per type from this frame and return it."""
    df = df.copy()
    if 'surprise_raw' not in df.columns:
        print(f"  [{label}] no surprise_raw column — keeping Excel surprise_std")
        return df, sigma_map
    if sigma_map is None:
        nc = df[~_covid_mask(df)]
        sigma_map = nc.groupby('type')['surprise_raw'].std()  # ddof=1
    df['surprise_std'] = df.apply(
        lambda r: (r['surprise_raw'] / sigma_map[r['type']])
        if (r['type'] in sigma_map.index and pd.notna(r['surprise_raw'])
            and sigma_map[r['type']] not in (0, np.nan)) else np.nan, axis=1)
    return df, sigma_map

# Main standardization: non-COVID sigma, computed once from EPU and shared so both
# frames use identical per-type sigma.
epu, SIGMA_NONCOVID = standardize_surprises(epu, None, 'EPU')
shut, _ = standardize_surprises(shut, SIGMA_NONCOVID, 'Shutdown')
print(f"  Non-COVID sigma computed for {len(SIGMA_NONCOVID)} types; "
      f"surprise_std recomputed uniformly in code.")

# Robustness variant: sigma over FULL sample including COVID
_epu_full_nc = epu.copy()
SIGMA_FULL = epu.groupby('type')['surprise_raw'].std()
epu['surprise_std_covid'] = epu.apply(
    lambda r: (r['surprise_raw'] / SIGMA_FULL[r['type']])
    if (r['type'] in SIGMA_FULL.index and pd.notna(r['surprise_raw'])
        and SIGMA_FULL[r['type']] not in (0, np.nan)) else np.nan, axis=1)
print(f"  Robustness variant surprise_std_covid (full-sample sigma) computed.")

# 3. COVID EXCLUSION
EXCLUDE_COVID = True   # confirmed by promotor

if EXCLUDE_COVID:
    for df, name in [(epu, 'EPU'), (shut, 'Shutdown')]:
        mask = (df['release_date'] >= '2020-04-01') & (df['release_date'] <= '2020-07-31')
        n = mask.sum()
        df.drop(df[mask].index, inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"  COVID exclusion ({name}): {n} rows dropped")
else:
    print("\n  COVID exclusion: OFF")

# 4. CLEAN DATA
print("\n" + "=" * 60)
print("SECTION 4: CLEANING")
print("=" * 60)

def clean(df, ret_col='ret_sym5'):
    before = len(df)
    df = df.dropna(subset=[ret_col, 'surprise_std']).copy()
    print(f"    Dropped {before - len(df)} rows -> {len(df)} remaining")
    return df

print("  EPU (ES):")
epu_clean = clean(epu)
print("  Shutdown (ES):")
shut_clean = clean(shut)


for df in [epu_clean, shut_clean]:
    df['year']       = df['release_date'].dt.year
    df['year_month'] = df['release_date'].dt.to_period('M').astype(str)

# Standardize FPU over estimation sample
if HAS_FPU:
    fpu_mean = epu_clean['FPU_raw'].mean()
    fpu_std_val = epu_clean['FPU_raw'].std()
    if pd.isna(fpu_std_val) or fpu_std_val == 0:
        HAS_FPU = False
        print("  WARNING: FPU_raw has zero or invalid SD — FPU models skipped")
    else:
        epu_clean['FPU_std'] = (epu_clean['FPU_raw'] - fpu_mean) / fpu_std_val
        n_valid = epu_clean['FPU_std'].notna().sum()
        print(f"\n  FPU standardized: mean={fpu_mean:.1f}, SD={fpu_std_val:.1f}, "
              f"valid obs={n_valid}/{len(epu_clean)}")

        # Lagged FPU at month level — addresses look-ahead concern raised in peer review
        fpu_monthly = (epu_clean[['year_month', 'FPU_std']]
                       .drop_duplicates('year_month')
                       .sort_values('year_month')
                       .copy())
        fpu_monthly['FPU_std_lag'] = fpu_monthly['FPU_std'].shift(1)
        epu_clean = epu_clean.merge(fpu_monthly[['year_month', 'FPU_std_lag']],
                                    on='year_month', how='left')
        n_lag_valid = epu_clean['FPU_std_lag'].notna().sum()
        print(f"  FPU lagged (t-1 month): valid obs={n_lag_valid}/{len(epu_clean)}")


# 4b. EVENT-TIME RESTRUCTURING  (shared-window de-duplication)
# Tag event_id
print("\n" + "=" * 60)
print("SECTION 4b: EVENT-TIME RESTRUCTURING")
print("=" * 60)

RET_COLS_ES = [c for c in ['ret_sym5','ret_10min','ret_15min','ret_30min','ret_sym10']
               if c in epu_clean.columns]
RET_COLS_ZN = [c for c in ['zn_sym5','zn_10min','zn_15min','zn_30min','zn_sym10']
               if c in epu_clean.columns]
ALL_RET_EPU = RET_COLS_ES + RET_COLS_ZN
ALL_RET_SHUT = [c for c in ['ret_sym5','ret_10min','ret_15min','ret_30min','ret_sym10',
                            'zn_sym5','zn_10min','zn_15min','zn_30min','zn_sym10']
                if c in shut_clean.columns]

epu_clean = add_event_id(epu_clean)
shut_clean = add_event_id(shut_clean)
print(f"  EPU:      {len(epu_clean)} rows -> {epu_clean['event_id'].nunique()} event-times")
print(f"  Shutdown: {len(shut_clean)} rows -> {shut_clean['event_id'].nunique()} event-times")
verify_shared_returns(epu_clean, ['ret_sym5'], 'EPU')
verify_shared_returns(shut_clean, ['ret_sym5'], 'Shutdown')
epu_clean = dedup_returns(epu_clean, ALL_RET_EPU)
shut_clean = dedup_returns(shut_clean, ALL_RET_SHUT)

epu_carry = [c for c in ['FPU_std','FPU_std_lag','fiscal_stress','year','year_month']
             if c in epu_clean.columns]
epu_wide, TOKEN_MAP = make_wide(epu_clean, ret_cols=ALL_RET_EPU, carry_cols=epu_carry)
shut_carry = [c for c in ['shutdown','pre_window','post_window','shutdown_id',
                          'year','year_month'] if c in shut_clean.columns]
shut_wide, TOKEN_MAP_SHUT = make_wide(shut_clean, ret_cols=ALL_RET_SHUT,
                                      carry_cols=shut_carry)
print(f"  EPU wide:      {len(epu_wide)} event-rows, "
      f"{sum(c.startswith('S_') for c in epu_wide.columns)} surprise columns")
print(f"  Shutdown wide: {len(shut_wide)} event-rows")

# Tokens present (non-empty) in each frame — used to build formulas without
# all-zero columns from types absent in a subsample.
def present_toks(wide_df, token_map):
    return [t for t in token_map.values()
            if f'has_{t}' in wide_df.columns and wide_df[f'has_{t}'].sum() > 0]


# 5. DESCRIPTIVE DIAGNOSTICS
if RUN_DESCRIPTIVES:
    print("\n" + "=" * 60)
    print("SECTION 5: DESCRIPTIVE DIAGNOSTICS")
    print("=" * 60)

    print("\nEvents per announcement type:")
    print(epu_clean['type'].value_counts().to_string())

    print(f"\nFiscal stress dummy: {epu_clean['fiscal_stress'].sum()} stressed / "
          f"{(epu_clean['fiscal_stress']==0).sum()} non-stressed "
          f"({epu_clean['fiscal_stress'].mean()*100:.1f}%)")

    fs_cross = pd.crosstab(epu_clean['type'], epu_clean['fiscal_stress'])
    fs_cross.columns = ['Non-stress', 'Fiscal stress']
    print("\nFiscal stress obs per type:")
    print(fs_cross.to_string())

    mean_ret  = epu_clean.groupby(['type','fiscal_stress'])['ret_sym5'].agg(['count','mean','std']).round(3)
    mean_surp = epu_clean.groupby(['type','fiscal_stress'])['surprise_std'].agg(['count','mean','std']).round(3)

    print(f"\nShutdown obs: {shut_clean['shutdown'].sum()} active, "
          f"{shut_clean['pre_window'].sum()} pre (20d), "
          f"{shut_clean['post_window'].sum()} post (45d), "
          f"{shut_clean['delayed_release'].sum()} delayed")

    print("\nES return summary (bps):")
    for col in ['ret_sym5','ret_10min','ret_15min','ret_30min','ret_sym10']:
        s = epu_clean[col].dropna()
        print(f"  {col}: mean={s.mean():.2f}, std={s.std():.2f}, "
              f"min={s.min():.2f}, max={s.max():.2f}")

    if HAS_ZN:
        print("\nZN return summary (bps):")
        for col in ['zn_sym5','zn_10min','zn_15min','zn_30min','zn_sym10']:
            if col in epu_clean.columns:
                s = epu_clean[col].dropna()
                print(f"  {col}: mean={s.mean():.2f}, std={s.std():.2f}, "
                      f"min={s.min():.2f}, max={s.max():.2f}")

# 6. REGRESSION HELPERS
def run_regression(df, formula, label="", cluster_col=None):
    try:
        if cluster_col is not None:
            model = smf.ols(formula=formula, data=df).fit(
                cov_type='cluster', cov_kwds={'groups': df[cluster_col]})
        else:
            model = smf.ols(formula=formula, data=df).fit(cov_type='HC3')
        return model
    except Exception as e:
        print(f"  ERROR in {label}: {e}")
        return None

def results_to_df(model, label=""):
    if model is None:
        return pd.DataFrame()
    df = pd.DataFrame({
        'Coefficient (bps)': model.params,
        'Std Error': model.bse,
        't-stat': model.tvalues,
        'p-value': model.pvalues,
        'CI Lower 95%': model.conf_int()[0],
        'CI Upper 95%': model.conf_int()[1],
    })
    df['Sig'] = df['p-value'].apply(
        lambda p: '***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.10 else '')))
    df['N'] = model.nobs
    df['R2'] = model.rsquared
    df['Adj R2'] = model.rsquared_adj
    df['Model'] = label
    return df

def stars(p):
    return '***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.10 else ''))

def run_wide_model(wide_df, dep, token_map, regime, label="", cluster_col='event_id'):
    """Estimate a joint type-specific model on the WIDE frame.
    Drops rows where dep or the regime variable is missing, restricts to types
    actually present, builds the formula, and clusters by event-time by default
    (set cluster_col=None for HC3). 'regime' is 'FPU_std', 'FPU_std_lag',
    'fiscal_stress', 'post_window', or 'pre_window'."""
    need = [dep] + ([regime] if regime else [])
    df = wide_df.dropna(subset=[c for c in need if c in wide_df.columns]).copy()
    if cluster_col:
        df = df.reset_index(drop=True)
    toks = present_toks(df, token_map)
    formula = build_wide_formula(dep, token_map, regime=regime, present_tokens=toks)
    try:
        if cluster_col is not None:
            m = smf.ols(formula, data=df).fit(cov_type='cluster',
                                              cov_kwds={'groups': df[cluster_col]})
        else:
            m = smf.ols(formula, data=df).fit(cov_type='HC3')
        return m
    except Exception as e:
        print(f"  ERROR in {label}: {e}")
        return None

def print_wide_interactions(model, regime, threshold=0.10):
    """Print significant per-type interaction terms S_<token>:<regime>."""
    if model is None:
        print("      (model failed)")
        return
    found = False
    for param, coef in model.params.items():
        if param.startswith('S_') and param.endswith(f':{regime}'):
            p = model.pvalues[param]
            if p < threshold:
                ty = param[2:].rsplit(f':{regime}', 1)[0]
                print(f"      {ty}: {coef:.4f} bps {stars(p)} (p={p:.4f})")
                found = True
    if not found:
        print(f"      None at p<{threshold}")

def print_type_interactions(model, interact_key, threshold=0.10):
    found = False
    for param, coef in model.params.items():
        if interact_key in param and 'surprise_std' in param and param != interact_key:
            p = model.pvalues[param]
            if p < threshold:
                print(f"      {param}: {coef:.4f} bps {stars(p)} (p={p:.4f})")
                found = True
    if not found:
        print(f"      None at p<{threshold}")

# Model formula strings
M4_DUMMY_FORMULA = ('ret_sym5 ~ C(type) + fiscal_stress + '
                    'C(type):surprise_std + C(type):surprise_std:fiscal_stress')
M4_CONT_FORMULA  = ('ret_sym5 ~ C(type) + FPU_std + '
                    'C(type):surprise_std + C(type):surprise_std:FPU_std')

all_results = []

# 7a
if RUN_EPU_REGRESSIONS:
    print("\n" + "=" * 60)
    print("SECTION 7a: CONTINUOUS FPU — MAIN SPECIFICATION (ES)")
    print("=" * 60)

    epu_clean = epu_clean.reset_index(drop=True)
    epu_clean['surprise_abs'] = epu_clean['surprise_std'].abs()
    epu_clean['ret_sym5_abs'] = epu_clean['ret_sym5'].abs()

    # MC1: Baseline |return| ~ |surprise| (long, event-clustered)
    mc1 = run_regression(epu_clean, 'ret_sym5_abs ~ surprise_abs', 'MC1',
                         cluster_col='event_id')
    all_results.append(results_to_df(mc1, 'MC1: Baseline ES'))
    p1 = mc1.pvalues.get('surprise_abs', np.nan)
    print(f"\n  MC1 Baseline: N={int(mc1.nobs)}, R2={mc1.rsquared:.4f}")
    print(f"    beta={mc1.params.get('surprise_abs',np.nan):.4f} bps, p={p1:.4f} {stars(p1)}")
    print(f"    1-SD surprise -> {abs(mc1.params.get('surprise_abs',0)):.2f} bps additional move")

    if HAS_FPU:
        # Drop NaN FPU rows once, reset index for clustered SE compatibility
        epu_fpu_clean = epu_clean.dropna(subset=['FPU_std']).copy().reset_index(drop=True)
        print(f"  FPU valid sample: N={len(epu_fpu_clean)}")
        # MC2: Pooled continuous FPU interaction (absolute return/surprise, long,
        # event-clustered) — magnitude model, consistent with MC1 and methodology.
        mc2 = run_regression(epu_fpu_clean,
            'ret_sym5_abs ~ surprise_abs + FPU_std + surprise_abs:FPU_std', 'MC2',
            cluster_col='event_id')
        all_results.append(results_to_df(mc2, 'MC2: Continuous FPU pooled'))
        p2c = mc2.pvalues.get('surprise_abs:FPU_std', np.nan)
        print(f"\n  MC2 Pooled continuous FPU: N={int(mc2.nobs)}, R2={mc2.rsquared:.4f}")
        print(f"    beta={mc2.params.get('surprise_abs:FPU_std',np.nan):.4f} bps/SD, "
              f"p={p2c:.4f} {stars(p2c)}")

        # MC3: + type FE (absolute return/surprise, long, event-clustered)
        mc3 = run_regression(epu_fpu_clean,
            'ret_sym5_abs ~ surprise_abs + FPU_std + surprise_abs:FPU_std + C(type)', 'MC3',
            cluster_col='event_id')
        all_results.append(results_to_df(mc3, 'MC3: Continuous FPU + TypeFE'))
        p3c = mc3.pvalues.get('surprise_abs:FPU_std', np.nan)
        print(f"\n  MC3 + Type FE: N={int(mc3.nobs)}, R2={mc3.rsquared:.4f}")
        print(f"    beta={mc3.params.get('surprise_abs:FPU_std',np.nan):.4f} bps, "
              f"p={p3c:.4f} {stars(p3c)}")

        # MC4: MAIN MODEL — joint type-specific slopes + FPU interaction
        mc4 = run_wide_model(epu_wide, 'ret_sym5', TOKEN_MAP, 'FPU_std',
                             'MC4_main', cluster_col='event_id')
        all_results.append(results_to_df(mc4, 'MC4: Main Continuous FPU (wide)'))
        if mc4 is not None:
            print(f"\n  MC4 MAIN CONTINUOUS FPU (wide, event-clustered): "
                  f"N={int(mc4.nobs)}, R2={mc4.rsquared:.4f}")
            print("    Type-specific FPU interactions (bps per 1-SD FPU increase):")
            print_wide_interactions(mc4, 'FPU_std')

        # MC5: + year FE (wide). Year dummies via C(year) on the wide frame.
        mc5_formula = (build_wide_formula('ret_sym5', TOKEN_MAP, regime='FPU_std',
                       present_tokens=present_toks(
                           epu_wide.dropna(subset=['FPU_std']), TOKEN_MAP))
                       + ' + C(year)')
        _epu_wide_fpu = epu_wide.dropna(subset=['FPU_std']).reset_index(drop=True)
        try:
            mc5 = smf.ols(mc5_formula, data=_epu_wide_fpu).fit(
                cov_type='cluster', cov_kwds={'groups': _epu_wide_fpu['event_id']})
        except Exception as e:
            print(f"  ERROR in MC5: {e}"); mc5 = None
        all_results.append(results_to_df(mc5, 'MC5: Continuous FPU + YearFE (wide)'))
        if mc5 is not None:
            print(f"\n  MC5 + Year FE (wide): N={int(mc5.nobs)}, R2={mc5.rsquared:.4f}")

        # MC4 — HC3 variant (robustness vs the event-clustered main)
        mc4_cluster = run_wide_model(epu_wide, 'ret_sym5', TOKEN_MAP, 'FPU_std',
                                     'MC4_hc3', cluster_col=None)
        all_results.append(results_to_df(mc4_cluster, 'MC4: Continuous FPU HC3 (wide)'))
        if mc4_cluster is not None:
            print(f"\n  MC4 HC3 SE (wide): N={int(mc4_cluster.nobs)}, "
                  f"R2={mc4_cluster.rsquared:.4f}")
            print("    Significant interactions:")
            print_wide_interactions(mc4_cluster, 'FPU_std')
    else:
        print("  SKIPPED: FPU_raw not available")
        mc2 = mc3 = mc4 = mc5 = mc4_cluster = None

    # ZN continuous FPU models (if ZN available)
    if HAS_ZN and HAS_FPU and 'zn_sym5' in epu_clean.columns:
        print("\n  --- Continuous FPU models: ZN (10-year Treasury) ---")
        epu_zn = epu_clean.dropna(subset=['zn_sym5', 'surprise_std']).copy()
        epu_zn['zn_sym5_abs'] = epu_zn['zn_sym5'].abs()
        epu_zn['surprise_abs'] = epu_zn['surprise_std'].abs()

        mc1_zn = run_regression(epu_zn, 'zn_sym5_abs ~ surprise_abs', 'MC1_ZN')
        all_results.append(results_to_df(mc1_zn, 'MC1: Baseline ZN'))
        p1z = mc1_zn.pvalues.get('surprise_abs', np.nan)
        print(f"\n  MC1 ZN Baseline: N={int(mc1_zn.nobs)}, R2={mc1_zn.rsquared:.4f}")
        print(f"    beta={mc1_zn.params.get('surprise_abs',np.nan):.4f} bps, "
              f"p={p1z:.4f} {stars(p1z)}")

        # MC4 ZN MAIN — joint type-specific (WIDE), event-clustered
        mc4_zn = run_wide_model(epu_wide, 'zn_sym5', TOKEN_MAP, 'FPU_std',
                                'MC4_ZN', cluster_col='event_id')
        all_results.append(results_to_df(mc4_zn, 'MC4: Main Continuous FPU ZN (wide)'))
        if mc4_zn is not None:
            print(f"\n  MC4 ZN MAIN (wide, event-clustered): N={int(mc4_zn.nobs)}, "
                  f"R2={mc4_zn.rsquared:.4f}")
            print("    Type-specific FPU interactions (ZN, bps per 1-SD FPU):")
            print_wide_interactions(mc4_zn, 'FPU_std')

        # MC4 ZN — HC3 variant
        mc4_zn_cluster = run_wide_model(epu_wide, 'zn_sym5', TOKEN_MAP, 'FPU_std',
                                        'MC4_ZN_hc3', cluster_col=None)
        all_results.append(results_to_df(mc4_zn_cluster, 'MC4: ZN HC3 SE (wide)'))
        if mc4_zn_cluster is not None:
            print(f"\n  MC4 ZN HC3 SE (wide): N={int(mc4_zn_cluster.nobs)}, "
                  f"R2={mc4_zn_cluster.rsquared:.4f}")
            print("    Significant interactions:")
            print_wide_interactions(mc4_zn_cluster, 'FPU_std')

# 7b. DUMMY-BASED EPU REGRESSIONS — ROBUSTNESS
if RUN_EPU_REGRESSIONS:
    print("\n" + "=" * 60)
    print("SECTION 7b: DUMMY-BASED EPU — ROBUSTNESS (ES)")
    print("=" * 60)

    # M2: Pooled dummy (long, event-clustered)
    m2 = run_regression(epu_clean,
        'ret_sym5 ~ surprise_std + fiscal_stress + surprise_std:fiscal_stress', 'M2',
        cluster_col='event_id')
    all_results.append(results_to_df(m2, 'M2: Dummy pooled'))
    p2 = m2.pvalues.get('surprise_std:fiscal_stress', np.nan)
    print(f"\n  M2 Dummy pooled: N={int(m2.nobs)}, R2={m2.rsquared:.4f}")
    print(f"    beta={m2.params.get('surprise_std:fiscal_stress',np.nan):.4f} bps, "
          f"p={p2:.4f} {stars(p2)}")

    # M3: + type FE (long, event-clustered)
    m3 = run_regression(epu_clean,
        'ret_sym5 ~ surprise_std + fiscal_stress + surprise_std:fiscal_stress + C(type)', 'M3',
        cluster_col='event_id')
    all_results.append(results_to_df(m3, 'M3: Dummy + TypeFE'))
    p3 = m3.pvalues.get('surprise_std:fiscal_stress', np.nan)
    print(f"\n  M3 + Type FE: N={int(m3.nobs)}, R2={m3.rsquared:.4f}")
    print(f"    beta={m3.params.get('surprise_std:fiscal_stress',np.nan):.4f} bps, "
          f"p={p3:.4f} {stars(p3)}")

    # M4: Dummy preferred — joint type-specific (WIDE), event-clustered
    m4 = run_wide_model(epu_wide, 'ret_sym5', TOKEN_MAP, 'fiscal_stress',
                        'M4_dummy', cluster_col='event_id')
    all_results.append(results_to_df(m4, 'M4: Dummy preferred (wide)'))
    if m4 is not None:
        print(f"\n  M4 Dummy preferred (wide, event-clustered): "
              f"N={int(m4.nobs)}, R2={m4.rsquared:.4f}")
        print("    Type-specific interactions:")
        print_wide_interactions(m4, 'fiscal_stress')

    # M5: + year FE (wide)
    _toks_d = present_toks(epu_wide.dropna(subset=['fiscal_stress']), TOKEN_MAP)
    m5_formula = (build_wide_formula('ret_sym5', TOKEN_MAP, regime='fiscal_stress',
                  present_tokens=_toks_d) + ' + C(year)')
    _epu_wide_d = epu_wide.dropna(subset=['fiscal_stress']).reset_index(drop=True)
    try:
        m5 = smf.ols(m5_formula, data=_epu_wide_d).fit(
            cov_type='cluster', cov_kwds={'groups': _epu_wide_d['event_id']})
    except Exception as e:
        print(f"  ERROR in M5: {e}"); m5 = None
    all_results.append(results_to_df(m5, 'M5: Dummy + YearFE (wide)'))
    if m5 is not None:
        print(f"\n  M5 + Year FE (wide): N={int(m5.nobs)}, R2={m5.rsquared:.4f}")

    # M4 — HC3 variant
    m4_cluster = run_wide_model(epu_wide, 'ret_sym5', TOKEN_MAP, 'fiscal_stress',
                                'M4_hc3', cluster_col=None)
    all_results.append(results_to_df(m4_cluster, 'M4: Dummy HC3 SE (wide)'))
    if m4_cluster is not None:
        print(f"\n  M4 HC3 SE (wide): N={int(m4_cluster.nobs)}, R2={m4_cluster.rsquared:.4f}")
        print("    Significant interactions:")
        print_wide_interactions(m4_cluster, 'fiscal_stress')

    # ZN dummy EPU models (wide)
    if HAS_ZN and 'zn_sym5' in epu_clean.columns:
        print("\n  --- Dummy EPU models: ZN (wide) ---")
        m4_zn = run_wide_model(epu_wide, 'zn_sym5', TOKEN_MAP, 'fiscal_stress',
                               'M4_ZN_dummy', cluster_col='event_id')
        all_results.append(results_to_df(m4_zn, 'M4: ZN Dummy preferred (wide)'))
        if m4_zn is not None:
            print(f"\n  M4 ZN Dummy preferred (wide): N={int(m4_zn.nobs)}, "
                  f"R2={m4_zn.rsquared:.4f}")
            print("    Type-specific interactions:")
            print_wide_interactions(m4_zn, 'fiscal_stress')

        m4_zn_cluster = run_wide_model(epu_wide, 'zn_sym5', TOKEN_MAP, 'fiscal_stress',
                                       'M4_ZN_hc3', cluster_col=None)
        all_results.append(results_to_df(m4_zn_cluster, 'M4: ZN Dummy HC3 SE (wide)'))
        if m4_zn_cluster is not None:
            print(f"\n  M4 ZN HC3 SE (wide): N={int(m4_zn_cluster.nobs)}, "
                  f"R2={m4_zn_cluster.rsquared:.4f}")
            print("    Significant interactions:")
            print_wide_interactions(m4_zn_cluster, 'fiscal_stress')

# 8. SHUTDOWN MODELS (ES — main; ZN added where applicable)
if RUN_SHUTDOWN_MODELS:
    print("\n" + "=" * 60)
    print("SECTION 8: SHUTDOWN MODELS (ES)")
    print("=" * 60)

    shut_clean = shut_clean.reset_index(drop=True)
    shut_clean['surprise_abs'] = shut_clean['surprise_std'].abs()
    shut_clean['ret_sym5_abs'] = shut_clean['ret_sym5'].abs()

    # M6: Active shutdown — pooled, absolute
    m6 = run_regression(shut_clean,
        'ret_sym5_abs ~ surprise_abs + shutdown + surprise_abs:shutdown', 'M6')
    all_results.append(results_to_df(m6, 'M6: Shutdown'))
    p6 = m6.pvalues.get('surprise_abs:shutdown', np.nan)
    print(f"\n  M6 Active ({shut_clean['shutdown'].sum()} obs): "
          f"beta={m6.params.get('surprise_abs:shutdown',np.nan):.4f} bps, "
          f"p={p6:.4f} {stars(p6)}")

    # M7: + type FE
    m7 = run_regression(shut_clean,
        'ret_sym5 ~ C(type) + shutdown + C(type):surprise_std + '
        'C(type):surprise_std:shutdown', 'M7')
    all_results.append(results_to_df(m7, 'M7: Shutdown + TypeFE'))
    print(f"\n  M7 Shutdown + Type FE: N={int(m7.nobs)}, R2={m7.rsquared:.4f}")

    # M8: Pre-window (20d)
    m8 = run_regression(shut_clean,
        'ret_sym5_abs ~ surprise_abs + pre_window + surprise_abs:pre_window', 'M8')
    all_results.append(results_to_df(m8, 'M8: Pre-window'))
    p8 = m8.pvalues.get('surprise_abs:pre_window', np.nan)
    print(f"\n  M8 Pre-window ({shut_clean['pre_window'].sum()} obs): "
          f"beta={m8.params.get('surprise_abs:pre_window',np.nan):.4f} bps, "
          f"p={p8:.4f} {stars(p8)}")

    # M9: Post-window (45d)
    m9 = run_regression(shut_clean,
        'ret_sym5_abs ~ surprise_abs + post_window + surprise_abs:post_window', 'M9')
    all_results.append(results_to_df(m9, 'M9: Post-window'))
    p9 = m9.pvalues.get('surprise_abs:post_window', np.nan)
    print(f"\n  M9 Post-window ({shut_clean['post_window'].sum()} obs): "
          f"beta={m9.params.get('surprise_abs:post_window',np.nan):.4f} bps, "
          f"p={p9:.4f} {stars(p9)}")

    # Combined episode dummy
    shut_clean['shutdown_episode'] = (
        (shut_clean['shutdown'] == 1) | (shut_clean['pre_window'] == 1) |
        (shut_clean['post_window'] == 1)).astype(int)
    n_ep = shut_clean['shutdown_episode'].sum()

    # M10: Combined episode
    m10 = run_regression(shut_clean,
        'ret_sym5_abs ~ surprise_abs + shutdown_episode + surprise_abs:shutdown_episode',
        'M10')
    all_results.append(results_to_df(m10, 'M10: Episode Combined'))
    p10 = m10.pvalues.get('surprise_abs:shutdown_episode', np.nan)
    print(f"\n  M10 Episode (N={n_ep}): "
          f"beta={m10.params.get('surprise_abs:shutdown_episode',np.nan):.4f} bps, "
          f"p={p10:.4f} {stars(p10)}")

    # M11: All three phases
    m11 = run_regression(shut_clean,
        'ret_sym5_abs ~ surprise_abs + shutdown + pre_window + post_window '
        '+ surprise_abs:shutdown + surprise_abs:pre_window + surprise_abs:post_window',
        'M11')
    all_results.append(results_to_df(m11, 'M11: Mechanisms'))
    print(f"\n  M11 Mechanisms:")
    for param in ['surprise_abs:shutdown','surprise_abs:pre_window','surprise_abs:post_window']:
        if param in m11.params:
            p = m11.pvalues[param]
            print(f"    {param}: {m11.params[param]:.4f} bps {stars(p)} (p={p:.4f})")

    # M12: Episode + type FE
    m12 = run_regression(shut_clean,
        'ret_sym5 ~ C(type) + shutdown_episode + '
        'C(type):surprise_std + C(type):surprise_std:shutdown_episode', 'M12')
    all_results.append(results_to_df(m12, 'M12: Episode + TypeFE'))
    print(f"\n  M12 Episode + Type FE: N={int(m12.nobs)}, R2={m12.rsquared:.4f}")
    sig12 = [(p, c) for p, c in m12.params.items()
             if 'shutdown_episode' in p and 'surprise_std' in p and m12.pvalues[p] < 0.10]
    if sig12:
        for param, coef in sig12:
            p = m12.pvalues[param]
            print(f"    {param}: {coef:.4f} bps {stars(p)} (p={p:.4f})")
    else:
        print("    No type-specific interactions at p<0.10")

    # M13: Post-window + type FE — MAIN SHUTDOWN RESULT (WIDE, event-clustered)
    m13 = run_wide_model(shut_wide, 'ret_sym5', TOKEN_MAP_SHUT, 'post_window',
                         'M13', cluster_col='event_id')
    all_results.append(results_to_df(m13, 'M13: Post-window + TypeFE (wide)'))
    if m13 is not None:
        print(f"\n  M13 Post-window + Type FE (45d, wide): N={int(m13.nobs)}, "
              f"R2={m13.rsquared:.4f}")
        print("    Significant (p<0.10):")
        print_wide_interactions(m13, 'post_window')

    # M13_pre: Pre-window + type FE — anticipation effects
    m13_pre = run_wide_model(shut_wide, 'ret_sym5', TOKEN_MAP_SHUT, 'pre_window',
                             'M13_pre', cluster_col='event_id')
    all_results.append(results_to_df(m13_pre, 'M13_pre: Pre-window + TypeFE (wide)'))
    if m13_pre is not None:
        print(f"\n  M13_pre Pre-window + Type FE (20d, wide): N={int(m13_pre.nobs)}, "
              f"R2={m13_pre.rsquared:.4f}")
        print("    Significant (p<0.10):")
        print_wide_interactions(m13_pre, 'pre_window')

    # M13b: Verified delayed releases only — stays LONG
    n_delayed = shut_clean['delayed_release'].sum()
    m13b = run_regression(shut_clean,
        'ret_sym5 ~ C(type) + delayed_release + '
        'C(type):surprise_std + C(type):surprise_std:delayed_release', 'M13b',
        cluster_col='event_id')
    all_results.append(results_to_df(m13b, 'M13b: Delayed Release + TypeFE'))
    print(f"\n  M13b Delayed (N={n_delayed}): N={int(m13b.nobs)}, R2={m13b.rsquared:.4f}")
    print("    Significant (p<0.10, excl. degenerate):")
    found = False
    for param, coef in m13b.params.items():
        if 'delayed_release' in param and 'surprise_std' in param:
            tn = param.replace('C(type)[T.','').replace(']:surprise_std:delayed_release','')
            if shut_clean[shut_clean['type'] == tn]['delayed_release'].sum() == 0:
                continue
            p = m13b.pvalues[param]
            if p < 0.10:
                print(f"      {param}: {coef:.4f} bps {stars(p)} (p={p:.4f})")
                found = True
    if not found:
        print("      None")

    # M13c: POOLED delayed-release model
    m13c = run_regression(shut_clean,
        'ret_sym5_abs ~ C(type) + surprise_abs + delayed_release + '
        'surprise_abs:delayed_release', 'M13c', cluster_col='event_id')
    all_results.append(results_to_df(m13c, 'M13c: Delayed Release Pooled'))
    if m13c is not None:
        lvl = m13c.params.get('delayed_release', np.nan)
        lvl_p = m13c.pvalues.get('delayed_release', np.nan)
        intr = m13c.params.get('surprise_abs:delayed_release', np.nan)
        intr_p = m13c.pvalues.get('surprise_abs:delayed_release', np.nan)
        print(f"\n  M13c Delayed POOLED (N={int(m13c.nobs)}, R2={m13c.rsquared:.4f}):")
        print(f"    delayed_release (level):          {lvl:.4f} bps {stars(lvl_p)} (p={lvl_p:.4f})")
        print(f"    surprise_abs:delayed (amplif.):   {intr:.4f} bps {stars(intr_p)} (p={intr_p:.4f})")

    # Phase summary
    shut_clean['phase'] = 'Non-episode'
    shut_clean.loc[shut_clean['pre_window'] == 1,  'phase'] = 'Pre-window (20d)'
    shut_clean.loc[shut_clean['shutdown'] == 1,    'phase'] = 'Active shutdown'
    shut_clean.loc[shut_clean['post_window'] == 1, 'phase'] = 'Post-window (45d)'
    phase_summary = shut_clean.groupby('phase')['ret_sym5'].agg(['count','mean','std']).round(3)
    print("\n  Mean ES ret [0,+5] (bps) by phase:")
    print(phase_summary.to_string())

    # ZN shutdown models (M13 equivalent)
    if HAS_ZN and 'zn_sym5' in shut_clean.columns:
        print("\n  --- Shutdown models: ZN ---")
        shut_zn = shut_clean.dropna(subset=['zn_sym5', 'surprise_std']).copy()
        shut_zn['zn_sym5_abs'] = shut_zn['zn_sym5'].abs()

        m6_zn = run_regression(shut_zn,
            'zn_sym5_abs ~ surprise_abs + shutdown + surprise_abs:shutdown', 'M6_ZN')
        all_results.append(results_to_df(m6_zn, 'M6: Shutdown ZN'))
        p6z = m6_zn.pvalues.get('surprise_abs:shutdown', np.nan)
        print(f"\n  M6 ZN: beta={m6_zn.params.get('surprise_abs:shutdown',np.nan):.4f} bps, "
              f"p={p6z:.4f} {stars(p6z)}")

        # M13c ZN: POOLED delayed-release model on ZN (absolute), parallel to ES M13c
        m13c_zn = run_regression(shut_zn,
            'zn_sym5_abs ~ C(type) + surprise_abs + delayed_release + '
            'surprise_abs:delayed_release', 'M13c_ZN', cluster_col='event_id')
        all_results.append(results_to_df(m13c_zn, 'M13c: Delayed Release Pooled ZN'))
        if m13c_zn is not None:
            lvlz = m13c_zn.params.get('delayed_release', np.nan)
            lvlz_p = m13c_zn.pvalues.get('delayed_release', np.nan)
            intrz = m13c_zn.params.get('surprise_abs:delayed_release', np.nan)
            intrz_p = m13c_zn.pvalues.get('surprise_abs:delayed_release', np.nan)
            print(f"\n  M13c Delayed POOLED ZN (N={int(m13c_zn.nobs)}, R2={m13c_zn.rsquared:.4f}):")
            print(f"    delayed_release (level):        {lvlz:.4f} bps {stars(lvlz_p)} (p={lvlz_p:.4f})")
            print(f"    surprise_abs:delayed:           {intrz:.4f} bps {stars(intrz_p)} (p={intrz_p:.4f})")

        m13_zn = run_wide_model(shut_wide, 'zn_sym5', TOKEN_MAP_SHUT, 'post_window',
                                'M13_ZN', cluster_col='event_id')
        all_results.append(results_to_df(m13_zn, 'M13: Post-window ZN (wide)'))
        if m13_zn is not None:
            print(f"\n  M13 ZN Post-window (wide): N={int(m13_zn.nobs)}, "
                  f"R2={m13_zn.rsquared:.4f}")
            print("    Significant (p<0.10):")
            print_wide_interactions(m13_zn, 'post_window')

        m13_pre_zn = run_wide_model(shut_wide, 'zn_sym5', TOKEN_MAP_SHUT, 'pre_window',
                                    'M13_pre_ZN', cluster_col='event_id')
        all_results.append(results_to_df(m13_pre_zn, 'M13_pre: Pre-window ZN (wide)'))
        if m13_pre_zn is not None:
            print(f"\n  M13_pre ZN Pre-window (wide): N={int(m13_pre_zn.nobs)}, "
                  f"R2={m13_pre_zn.rsquared:.4f}")
            print("    Significant (p<0.10):")
            print_wide_interactions(m13_pre_zn, 'pre_window')

# 9. PER-TYPE REGRESSIONS
if RUN_PER_TYPE:
    print("\n" + "=" * 60)
    print("SECTION 9: PER-TYPE REGRESSIONS")
    print("=" * 60)

    n_types = epu_clean['type'].nunique()
    bonf_thresh = 0.05 / n_types
    print(f"\n  {n_types} types. Bonferroni threshold: p < {bonf_thresh:.4f}")

    type_results = []

    if HAS_FPU:
        print("\n  --- Continuous FPU ---")
        for ann_type in sorted(epu_clean['type'].unique()):
            sub = epu_clean[epu_clean['type'] == ann_type].dropna(subset=['FPU_std']).copy()
            if len(sub) < 20 or sub['FPU_std'].std() == 0:
                continue
            m = run_regression(sub,
                'ret_sym5 ~ surprise_std + FPU_std + surprise_std:FPU_std', ann_type)
            if m is None:
                continue
            p_int  = m.pvalues.get('surprise_std:FPU_std', np.nan)
            p_bonf = min(p_int * n_types, 1.0)
            row = {'Type': ann_type, 'N': int(m.nobs), 'Spec': 'Continuous FPU',
                   'beta': round(m.params.get('surprise_std:FPU_std', np.nan), 4),
                   'p': round(p_int, 4), 'sig': stars(p_int),
                   'p_Bonf': round(p_bonf, 4), 'sig_Bonf': stars(p_bonf),
                   'R2': round(m.rsquared, 4)}
            type_results.append(row)
            bf = " <- Bonf sig" if p_bonf < 0.05 else ""
            print(f"  {ann_type:30s} beta={row['beta']:8.4f} "
                  f"{row['sig']:3s} (p={p_int:.4f}, Bonf={p_bonf:.4f}){bf}")

    print("\n  --- Dummy 75th pct ---")
    for ann_type in sorted(epu_clean['type'].unique()):
        sub = epu_clean[epu_clean['type'] == ann_type].copy()
        if len(sub) < 20 or sub['fiscal_stress'].std() == 0:
            continue
        m = run_regression(sub,
            'ret_sym5 ~ surprise_std + fiscal_stress + surprise_std:fiscal_stress', ann_type)
        if m is None:
            continue
        p_int  = m.pvalues.get('surprise_std:fiscal_stress', np.nan)
        p_bonf = min(p_int * n_types, 1.0)
        row = {'Type': ann_type, 'N': int(m.nobs), 'Spec': 'Dummy75',
               'N_stressed': int(sub['fiscal_stress'].sum()),
               'beta': round(m.params.get('surprise_std:fiscal_stress', np.nan), 4),
               'p': round(p_int, 4), 'sig': stars(p_int),
               'p_Bonf': round(p_bonf, 4), 'sig_Bonf': stars(p_bonf),
               'R2': round(m.rsquared, 4)}
        type_results.append(row)
        bf = " <- Bonf sig" if p_bonf < 0.05 else ""
        print(f"  {ann_type:30s} beta={row['beta']:8.4f} "
              f"{row['sig']:3s} (p={p_int:.4f}, Bonf={p_bonf:.4f}){bf}")

    type_results_df = pd.DataFrame(type_results)

# 9b. ASYMMETRY
if RUN_ASYMMETRY:
    print("\n" + "=" * 60)
    print("SECTION 9b: ASYMMETRY (DUMMY SPEC)")
    print("=" * 60)

    epu_asym = epu_clean[epu_clean['surprise_std'] != 0].copy()
    epu_asym['pos_surprise'] = (epu_asym['surprise_std'] > 0).astype(int)
    n_zero = (epu_clean['surprise_std'] == 0).sum()
    print(f"  Zeros dropped: {n_zero}, working N={len(epu_asym)}")

    ma1 = run_regression(epu_asym,
        'ret_sym5 ~ surprise_std + fiscal_stress + pos_surprise '
        '+ surprise_std:fiscal_stress + surprise_std:pos_surprise '
        '+ surprise_std:fiscal_stress:pos_surprise', 'MA1')
    all_results.append(results_to_df(ma1, 'MA1: Asymmetry Pooled'))
    if ma1:
        print(f"\n  MA1: N={int(ma1.nobs)}, R2={ma1.rsquared:.4f}")
        for param in ['surprise_std:fiscal_stress','surprise_std:pos_surprise',
                      'surprise_std:fiscal_stress:pos_surprise']:
            if param in ma1.params:
                p = ma1.pvalues[param]
                print(f"    {param}: {ma1.params[param]:.4f} bps {stars(p)} (p={p:.4f})")
        p3 = ma1.pvalues.get('surprise_std:fiscal_stress:pos_surprise', np.nan)
        c3 = ma1.params.get('surprise_std:fiscal_stress:pos_surprise', np.nan)
        if not np.isnan(p3):
            print(f"    -> Positive surprises amplified {'MORE' if c3>0 else 'LESS'} "
                  f"than negative during fiscal stress")

    ma2 = run_regression(epu_asym,
        'ret_sym5 ~ C(type) + fiscal_stress + pos_surprise + C(type):surprise_std '
        '+ surprise_std:fiscal_stress + surprise_std:pos_surprise '
        '+ surprise_std:fiscal_stress:pos_surprise', 'MA2')
    all_results.append(results_to_df(ma2, 'MA2: Asymmetry + TypeFE'))
    if ma2:
        print(f"\n  MA2 + Type FE: N={int(ma2.nobs)}, R2={ma2.rsquared:.4f}")
        for param in ['surprise_std:fiscal_stress','surprise_std:pos_surprise',
                      'surprise_std:fiscal_stress:pos_surprise']:
            if param in ma2.params:
                p = ma2.pvalues[param]
                print(f"    {param}: {ma2.params[param]:.4f} bps {stars(p)} (p={p:.4f})")

    print(f"\n  Split-sample per type:")
    print(f"  {'Type':30s} {'beta_pos':>10} {'p_pos':>8} {'beta_neg':>10} {'p_neg':>8}")
    print("  " + "-" * 65)
    asym_results = []
    for ann_type in sorted(epu_asym['type'].unique()):
        sub = epu_asym[epu_asym['type'] == ann_type].copy()
        pos_sub = sub[sub['pos_surprise'] == 1].copy()
        neg_sub = sub[sub['pos_surprise'] == 0].copy()
        if (len(pos_sub) < 15 or len(neg_sub) < 15 or
                pos_sub['fiscal_stress'].std() == 0 or neg_sub['fiscal_stress'].std() == 0):
            continue
        mp = run_regression(pos_sub,
            'ret_sym5 ~ surprise_std + fiscal_stress + surprise_std:fiscal_stress',
            f'{ann_type}_pos')
        mn = run_regression(neg_sub,
            'ret_sym5 ~ surprise_std + fiscal_stress + surprise_std:fiscal_stress',
            f'{ann_type}_neg')
        if mp is None or mn is None:
            continue
        b_pos = mp.params.get('surprise_std:fiscal_stress', np.nan)
        p_pos = mp.pvalues.get('surprise_std:fiscal_stress', np.nan)
        b_neg = mn.params.get('surprise_std:fiscal_stress', np.nan)
        p_neg = mn.pvalues.get('surprise_std:fiscal_stress', np.nan)
        opp = (b_pos > 0) != (b_neg > 0)
        flag = "YES" if (opp and ((p_pos < 0.10) or (p_neg < 0.10))) else ""
        print(f"  {ann_type:30s} {b_pos:10.4f} {stars(p_pos):3s} {p_pos:5.3f}  "
              f"{b_neg:10.4f} {stars(p_neg):3s} {p_neg:5.3f}  {flag}")
        asym_results.append({
            'Type': ann_type, 'N_pos': len(pos_sub), 'N_neg': len(neg_sub),
            'beta_pos': round(b_pos,4), 'p_pos': round(p_pos,4), 'sig_pos': stars(p_pos),
            'beta_neg': round(b_neg,4), 'p_neg': round(p_neg,4), 'sig_neg': stars(p_neg),
            'Asymmetric': flag,
        })
    asym_results_df = pd.DataFrame(asym_results)
    if not asym_results_df.empty and 'Asymmetric' in asym_results_df.columns:
        asym_found = asym_results_df[asym_results_df['Asymmetric'] != '']
    else:
        asym_found = pd.DataFrame()
    if len(asym_found) > 0:
        print(f"\n  Asymmetric types: {list(asym_found['Type'])}")
    else:
        print("\n  No asymmetric types detected")

# 10-11. ROBUSTNESS
if RUN_ROBUSTNESS:
    print("\n" + "=" * 60)
    print("SECTION 10: ROBUSTNESS — RETURN WINDOWS")
    print("=" * 60)

    window_results = []
    for w_label, ret_col in [
        ('5min [0,+5] MAIN', 'ret_sym5'), ('10min [0,+10]', 'ret_10min'),
        ('15min [0,+15]', 'ret_15min'), ('30min [0,+30]', 'ret_30min'),
        ('sym10 [-5,+5]', 'ret_sym10'),
    ]:
        if HAS_FPU:
            df_w = epu_clean.dropna(subset=[ret_col, 'surprise_std', 'FPU_std']).copy()
            formula = (f'{ret_col} ~ C(type) + FPU_std + '
                       f'C(type):surprise_std + C(type):surprise_std:FPU_std')
            ikey = 'FPU_std'
        else:
            df_w = epu_clean.dropna(subset=[ret_col, 'surprise_std']).copy()
            formula = (f'{ret_col} ~ C(type) + fiscal_stress + '
                       f'C(type):surprise_std + C(type):surprise_std:fiscal_stress')
            ikey = 'fiscal_stress'
        if len(df_w) < 50:
            continue
        m = run_regression(df_w, formula, f'win_{w_label}')
        if m is None:
            continue
        sig = sum(1 for p, v in m.pvalues.items()
                  if ikey in p and 'surprise_std' in p and p != ikey and v < 0.05)
        window_results.append({'Window': w_label, 'N': int(m.nobs),
                                'R2': round(m.rsquared,4), 'N_sig': sig})
        print(f"  {w_label:20s}: N={int(m.nobs)}, R2={m.rsquared:.4f}, {sig} sig (p<0.05)")
    window_results_df = pd.DataFrame(window_results)

    print("\n" + "=" * 60)
    print("SECTION 10b: ROBUSTNESS — LAGGED FPU SPECIFICATION")
    print("=" * 60)

    # Addresses look-ahead concern: contemporaneous monthly FPU includes newspaper
    if HAS_FPU and 'FPU_std_lag' in epu_wide.columns:
        n_lag = int(epu_wide['FPU_std_lag'].notna().sum())
        print(f"  Lagged FPU sample (wide): N={n_lag} event-times "
              f"(vs {int(epu_wide['FPU_std'].notna().sum())} contemporaneous)")

        # MC4 lagged — joint type-specific (WIDE), event-clustered
        mc4_lag = run_wide_model(epu_wide, 'ret_sym5', TOKEN_MAP, 'FPU_std_lag',
                                 'MC4_lag', cluster_col='event_id')
        all_results.append(results_to_df(mc4_lag, 'MC4: Lagged FPU (robustness, wide)'))
        if mc4_lag is not None:
            print(f"\n  MC4 Lagged FPU (wide, event-clustered): N={int(mc4_lag.nobs)}, "
                  f"R2={mc4_lag.rsquared:.4f}")
            print("    Significant interactions (p<0.10):")
            print_wide_interactions(mc4_lag, 'FPU_std_lag')

        # MC4 lagged — HC3 variant
        mc4_lag_cluster = run_wide_model(epu_wide, 'ret_sym5', TOKEN_MAP, 'FPU_std_lag',
                                         'MC4_lag_hc3', cluster_col=None)
        all_results.append(results_to_df(mc4_lag_cluster,
                                         'MC4: Lagged FPU HC3 (robustness, wide)'))
        if mc4_lag_cluster is not None:
            print(f"\n  MC4 Lagged FPU HC3 (wide): N={int(mc4_lag_cluster.nobs)}, "
                  f"R2={mc4_lag_cluster.rsquared:.4f}")
            print("    Significant interactions (p<0.10):")
            print_wide_interactions(mc4_lag_cluster, 'FPU_std_lag')
    else:
        print("  SKIPPED: FPU_raw not available or lag not constructed")
        mc4_lag = mc4_lag_cluster = None

    # SECTION 10c: ROBUSTNESS — SURPRISE STANDARDIZATION BASIS
    print("\n" + "=" * 60)
    print("SECTION 10c: ROBUSTNESS — SURPRISE STANDARDIZATION INCLUDING COVID")
    print("=" * 60)

    if HAS_FPU and 'surprise_std_covid' in epu_clean.columns:
        # Build a wide frame whose surprise columns use the COVID-sigma variant.
        epu_cov = epu_clean.copy()
        epu_cov['surprise_std'] = epu_cov['surprise_std_covid']
        epu_cov_wide, _ = make_wide(
            epu_cov, surprise_col='surprise_std', ret_cols=ALL_RET_EPU,
            carry_cols=[c for c in ['FPU_std','fiscal_stress','year','year_month']
                        if c in epu_cov.columns])
        mc4_cov = run_wide_model(epu_cov_wide, 'ret_sym5', TOKEN_MAP, 'FPU_std',
                                 'MC4_covid_sigma', cluster_col='event_id')
        all_results.append(results_to_df(mc4_cov, 'MC4: COVID-sigma standardization (robustness)'))
        if mc4_cov is not None:
            print(f"\n  MC4 with full-sample (incl. COVID) sigma: N={int(mc4_cov.nobs)}, "
                  f"R2={mc4_cov.rsquared:.4f}")
            print("    Significant interactions (p<0.10):")
            print_wide_interactions(mc4_cov, 'FPU_std')
            print("  Compare against the main MC4 above; stable signs/significance"
                  " indicate the COVID standardization basis does not drive results.")
    else:
        print("  SKIPPED: surprise_std_covid not available")

    print("\n" + "=" * 60)
    print("SECTION 11: ROBUSTNESS — WINSORIZED SAMPLE")
    print("=" * 60)

    def winsorize_col(s, lo=0.01, hi=0.99):
        return s.clip(s.quantile(lo), s.quantile(hi))

    ew = epu_clean.copy()
    ew['ret_sym5_w']     = winsorize_col(ew['ret_sym5'])
    ew['surprise_std_w'] = winsorize_col(ew['surprise_std'])

    if HAS_FPU:
        mw = run_regression(ew,
            'ret_sym5_w ~ C(type) + FPU_std + '
            'C(type):surprise_std_w + C(type):surprise_std_w:FPU_std', 'wins_cont')
        ikey_w = 'FPU_std'
    else:
        mw = run_regression(ew,
            'ret_sym5_w ~ C(type) + fiscal_stress + '
            'C(type):surprise_std_w + C(type):surprise_std_w:fiscal_stress', 'wins_dummy')
        ikey_w = 'fiscal_stress'

    if mw:
        all_results.append(results_to_df(mw, 'Winsorized model'))
        sig_w = sum(1 for p, v in mw.pvalues.items()
                    if ikey_w in p and 'surprise_std' in p and p != ikey_w and v < 0.05)
        print(f"  Winsorized: N={int(mw.nobs)}, R2={mw.rsquared:.4f}, {sig_w} sig")
        for param, coef in mw.params.items():
            if ikey_w in param and 'surprise_std' in param and param != ikey_w:
                p = mw.pvalues[param]
                print(f"    {param}: {coef:.4f} bps {stars(p)} (p={p:.4f})")

    print("\n" + "=" * 60)
    print("SECTION 11b: ROBUSTNESS — EPU THRESHOLD (70, 75, 80, 90th pct)")
    print("=" * 60)

    if HAS_FPU:
        epu_th = epu_clean.dropna(subset=['FPU_raw','ret_sym5','surprise_std']).copy()
        print(f"  Threshold sample N={len(epu_th)}")
        # Compute percentile cutoffs over the DISTINCT MONTHLY FPU values
        monthly_fpu = epu_th.drop_duplicates('year_month')['FPU_raw']
        thresh_results = []
        for pct in [70, 75, 80, 90]:
            thresh = monthly_fpu.quantile(pct / 100)
            epu_th[f'fs_{pct}'] = (epu_th['FPU_raw'] > thresh).astype(int)
            f_t = (f'ret_sym5 ~ C(type) + fs_{pct} + '
                   f'C(type):surprise_std + C(type):surprise_std:fs_{pct}')
            m_t = run_regression(epu_th, f_t, f'thresh_{pct}')
            if m_t is None:
                continue
            n_st = epu_th[f'fs_{pct}'].sum()
            sig_t = sum(1 for p, v in m_t.pvalues.items()
                        if f'fs_{pct}' in p and 'surprise_std' in p and v < 0.05)
            print(f"\n  {pct}th pct (cutoff={thresh:.1f}): N={int(m_t.nobs)}, "
                  f"stressed={n_st} ({n_st/len(epu_th)*100:.1f}%), "
                  f"R2={m_t.rsquared:.4f}, {sig_t} sig (p<0.05)")
            print("    Significant at p<0.10:")
            found_t = False
            for param, coef in m_t.params.items():
                if f'fs_{pct}' in param and 'surprise_std' in param:
                    p = m_t.pvalues[param]
                    if p < 0.10:
                        short = param
                        for pre in ('C(type)[T.', 'C(type)['):
                            if short.startswith(pre):
                                short = short[len(pre):]
                                break
                        short = short.split(']')[0].split(':')[0]
                        print(f"      {short}: {coef:.4f} bps {stars(p)} (p={p:.4f})")
                        found_t = True
            if not found_t:
                print("      None")
            thresh_results.append({'Threshold': f'{pct}th pct', 'Cutoff': round(thresh,1),
                                    'N_stressed': n_st,
                                    'Stressed_pct': round(n_st/len(epu_th)*100,1),
                                    'N_sig_p05': sig_t, 'R2': round(m_t.rsquared,4)})
            all_results.append(results_to_df(m_t, f'Threshold {pct}th pct'))
        thresh_results_df = pd.DataFrame(thresh_results)
    else:
        print("  SKIPPED: FPU_raw not available")
        thresh_results_df = pd.DataFrame()

# 12. SAVE RESULTS
if RUN_SAVE:
    print("\n" + "=" * 60)
    print("SECTION 12: SAVING")
    print("=" * 60)

    output_path = OUTPUT_DIR / 'regression_results.xlsx'
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

        if RUN_EPU_REGRESSIONS:
            if HAS_FPU:
                if mc2:        results_to_df(mc2,'MC2').to_excel(writer, sheet_name='MC2 Cont Pooled', index=True)
                if mc3:        results_to_df(mc3,'MC3').to_excel(writer, sheet_name='MC3 Cont TypeFE', index=True)
                if mc4:        results_to_df(mc4,'MC4').to_excel(writer, sheet_name='MC4 Main Cont', index=True)
                if mc4_cluster:results_to_df(mc4_cluster,'MC4C').to_excel(writer, sheet_name='MC4 Cont Clustered', index=True)
                if mc5:        results_to_df(mc5,'MC5').to_excel(writer, sheet_name='MC5 Cont YearFE', index=True)
                if 'mc4_lag' in globals() and mc4_lag:
                    results_to_df(mc4_lag, 'MC4_lag').to_excel(writer, sheet_name='MC4 Lagged FPU', index=True)
                if 'mc4_lag_cluster' in globals() and mc4_lag_cluster:
                    results_to_df(mc4_lag_cluster, 'MC4_lagC').to_excel(writer, sheet_name='MC4 Lagged Clustered', index=True)
                if HAS_ZN and 'mc4_zn' in globals() and mc4_zn:
                    results_to_df(mc4_zn,'MC4_ZN').to_excel(writer, sheet_name='MC4 ZN Main', index=True)
                if HAS_ZN and 'mc4_zn_cluster' in globals() and mc4_zn_cluster:
                    results_to_df(mc4_zn_cluster, 'MC4_ZNC').to_excel(writer, sheet_name='MC4 ZN Clustered', index=True)
                if HAS_ZN and 'm4_zn' in globals() and m4_zn:
                    results_to_df(m4_zn, 'M4_ZN').to_excel(writer, sheet_name='M4 ZN Dummy', index=True)
                if HAS_ZN and 'm4_zn_cluster' in globals() and m4_zn_cluster:
                    results_to_df(m4_zn_cluster, 'M4_ZNC').to_excel(writer, sheet_name='M4 ZN Clustered', index=True)
            results_to_df(m4,'M4').to_excel(writer, sheet_name='M4 Dummy', index=True)
            results_to_df(m4_cluster,'M4C').to_excel(writer, sheet_name='M4 Dummy Clustered', index=True)
            results_to_df(m5,'M5').to_excel(writer, sheet_name='M5 Dummy YearFE', index=True)

        if RUN_SHUTDOWN_MODELS:
            results_to_df(m6,'M6').to_excel(writer, sheet_name='M6 Active', index=True)
            results_to_df(m8,'M8').to_excel(writer, sheet_name='M8 Pre', index=True)
            results_to_df(m9,'M9').to_excel(writer, sheet_name='M9 Post', index=True)
            results_to_df(m10,'M10').to_excel(writer, sheet_name='M10 Episode', index=True)
            results_to_df(m11,'M11').to_excel(writer, sheet_name='M11 Mechanisms', index=True)
            results_to_df(m12,'M12').to_excel(writer, sheet_name='M12 Episode TypeFE', index=True)
            results_to_df(m13,'M13').to_excel(writer, sheet_name='M13 Post TypeFE', index=True)
            results_to_df(m13b,'M13b').to_excel(writer, sheet_name='M13b Delayed', index=True)
            if 'm13c' in globals() and m13c is not None:
                results_to_df(m13c,'M13c').to_excel(writer, sheet_name='M13c Delayed Pooled', index=True)
            results_to_df(m13_pre, 'M13pre').to_excel(writer, sheet_name='M13pre Pre TypeFE', index=True)
            if HAS_ZN and 'm13_zn' in globals() and m13_zn:
                results_to_df(m13_zn,'M13_ZN').to_excel(writer, sheet_name='M13 Post ZN', index=True)
            if 'm13c_zn' in globals() and m13c_zn is not None:
                results_to_df(m13c_zn,'M13c_ZN').to_excel(writer, sheet_name='M13c Delayed Pooled ZN', index=True)
            phase_summary.to_excel(writer, sheet_name='Phase Summary')

        if RUN_PER_TYPE and 'type_results_df' in globals():
            type_results_df.to_excel(writer, sheet_name='Per-Type Results', index=False)

        if RUN_ASYMMETRY:
            results_to_df(ma1,'MA1').to_excel(writer, sheet_name='MA1 Asymmetry', index=True)
            results_to_df(ma2,'MA2').to_excel(writer, sheet_name='MA2 Asym TypeFE', index=True)
            if 'asym_results_df' in globals() and not asym_results_df.empty:
                asym_results_df.to_excel(writer, sheet_name='Asymmetry Split', index=False)

        if RUN_ROBUSTNESS:
            window_results_df.to_excel(writer, sheet_name='Windows', index=False)
            if not thresh_results_df.empty:
                thresh_results_df.to_excel(writer, sheet_name='EPU Thresholds', index=False)

        if RUN_DESCRIPTIVES:
            fs_cross.to_excel(writer, sheet_name='FS Counts')
            mean_ret.to_excel(writer, sheet_name='Mean Returns')
            mean_surp.to_excel(writer, sheet_name='Mean Surprises')

        if all_results:
            pd.concat(all_results).to_excel(writer, sheet_name='All Coefficients', index=True)

        epu_clean.to_excel(writer, sheet_name='Data EPU', index=False)
        shut_clean.to_excel(writer, sheet_name='Data Shutdown', index=False)

        if RUN_DESCRIPTIVE_TABLES:
            # DESCRIPTIVE TABLES (Methodology)

            type_meta = {
                'NFP': ('Real activity', 'BLS', '8:30'),
                'Unemployment rate': ('Real activity', 'BLS', '8:30'),
                'Retail sales': ('Real activity', 'Census', '8:30'),
                'Industrial production': ('Real activity', 'FRB', '9:15'),
                'Consumer credit': ('Real activity', 'FRB', '15:00'),
                'Consumer confidence': ('Forward-looking', 'CB', '10:00'),
                'ISM PMI': ('Forward-looking', 'ISM', '10:00'),
                'ISM N-Mfg PMI': ('Forward-looking', 'ISM', '10:00'),
                'CPI': ('Prices', 'BLS', '8:30'),
                'New home sales': ('Consumption', 'Census', '10:00'),
                'Existing home sales': ('Consumption', 'NAR', '10:00'),
                'Durable goods': ('Investment', 'Census', '8:30'),
                'Construction spending': ('Investment', 'Census', '10:00'),
                'GDP': ('Output', 'BEA', '8:30'),
                'Federal budget': ('Government', 'FMS', '14:00'),
                'US International trade': ('Net exports', 'BEA', '8:30'),
            }
            t1_rows = []
            for ann_type, (category, source, time) in type_meta.items():
                sub = epu_clean[epu_clean['type'] == ann_type]
                n = len(sub)
                mean_abs_ret = round(sub['ret_sym5'].abs().mean(), 2) if n > 0 else np.nan
                t1_rows.append({
                    'Category': category,
                    'Announcement': ann_type,
                    'Source': source,
                    'Announcement time': time,
                    'N': n,
                    'Mean |ret| [0,+5] (bps)': mean_abs_ret,
                })
            table1 = pd.DataFrame(t1_rows).sort_values(['Category', 'Announcement'])
            table1.to_excel(writer, sheet_name='Table1 Announcements', index=False)

            if HAS_FPU:
                def fpu_bucket(x):
                    if pd.isna(x): return np.nan
                    if x < 0:   return '<0 SD'
                    if x < 1:   return '0–1 SD'
                    return '≥1 SD'


                epu_clean['fpu_bucket'] = epu_clean['FPU_std'].apply(fpu_bucket)
                t2_rows = []
                for ann_type in sorted(epu_clean['type'].unique()):
                    row = {'Announcement': ann_type}
                    for bucket in ['<0 SD', '0–1 SD', '≥1 SD']:
                        sub = epu_clean[(epu_clean['type'] == ann_type) &
                                        (epu_clean['fpu_bucket'] == bucket)]
                        row[f'N ({bucket})'] = len(sub)
                        row[f'Mean ret [0,+5] ({bucket})'] = round(sub['ret_sym5'].mean(), 2) if len(sub) > 0 else np.nan
                    t2_rows.append(row)
                # totals row
                tot = {'Announcement': 'Total'}
                for bucket in ['<0 SD', '0–1 SD', '≥1 SD']:
                    sub = epu_clean[epu_clean['fpu_bucket'] == bucket]
                    tot[f'N ({bucket})'] = len(sub)
                    tot[f'Mean ret [0,+5] ({bucket})'] = round(sub['ret_sym5'].mean(), 2) if len(sub) > 0 else np.nan
                t2_rows.append(tot)
                table2 = pd.DataFrame(t2_rows)
                table2.to_excel(writer, sheet_name='Table2 FPU Buckets', index=False)

            shutdown_dates = {
                1: ('1976-09-30', '1976-11-10'),
                2: ('1977-09-30', '1977-10-13'),
                3: ('1977-10-31', '1977-11-09'),
                4: ('1977-11-30', '1977-12-09'),
                5: ('1978-09-30', '1978-10-18'),
                6: ('1979-09-30', '1979-10-12'),
                7: ('1981-11-20', '1981-11-23'),
                8: ('1982-09-30', '1982-10-02'),
                9: ('1982-12-17', '1982-12-21'),
                10: ('1983-11-10', '1983-11-14'),
                11: ('1984-09-30', '1984-10-03'),
                12: ('1984-10-03', '1984-10-05'),
                13: ('1986-10-16', '1986-10-18'),
                14: ('1987-12-18', '1987-12-20'),
                15: ('1990-10-05', '1990-10-09'),
                16: ('1995-11-14', '1995-11-19'),
                17: ('1995-12-16', '1996-01-06'),
                18: ('2013-10-01', '2013-10-17'),
                19: ('2018-01-20', '2018-01-22'),
                20: ('2018-02-09', '2018-02-09'),
                21: ('2018-12-22', '2019-01-25'),
                22: ('2025-10-01', '2025-11-12'),
                23: ('2026-01-31', '2026-02-03'),
                24: ('2026-02-14', '2026-04-30'),
            }
            t3_rows = []
            for sid, (start, end) in shutdown_dates.items():
                sub = shut_clean[shut_clean['shutdown_id'] == sid]
                n_obs = len(sub[sub['shutdown'] == 1])
                n_types = sub[sub['shutdown'] == 1]['type'].nunique()
                start_dt = pd.to_datetime(start)
                end_dt = pd.to_datetime(end)
                days = (end_dt - start_dt).days + 1
                t3_rows.append({
                    'Shutdown ID': sid,
                    'Start': start_dt.strftime('%d/%m/%Y'),
                    'End': end_dt.strftime('%d/%m/%Y'),
                    'Days': days,
                    'N observations (active)': n_obs,
                    'N announcement types': n_types,
                })
            table3 = pd.DataFrame(t3_rows)
            table3.to_excel(writer, sheet_name='Table3 Shutdowns', index=False)

            phases = ['Pre-window (20d)', 'Active shutdown', 'Post-window (45d)']
            phase_col_map = {
                'Pre-window (20d)': 'pre_window',
                'Active shutdown': 'shutdown',
                'Post-window (45d)': 'post_window',
            }
            t4_data = {}
            for phase in phases:
                col = phase_col_map[phase]
                sub = shut_clean[shut_clean[col] == 1]
                on_time = len(sub[sub['delayed_release'] == 0])
                delayed = len(sub[sub['delayed_release'] == 1])
                total = len(sub)
                prop = round(delayed / total * 100, 1) if total > 0 else 0
                t4_data[phase] = {
                    'On time': on_time,
                    'Delayed': delayed,
                    'Total': total,
                    'Delayed (%)': prop,
                }
            table4 = pd.DataFrame(t4_data).T
            table4.index.name = 'Phase'
            table4.to_excel(writer, sheet_name='Table4 Release Timing')


            def summary_stats(df, ret_col, zn_col=None):
                row = {
                    f'ES mean (bps)': round(df[ret_col].mean(), 2),
                    f'ES std (bps)': round(df[ret_col].std(), 2),
                }
                if zn_col and zn_col in df.columns:
                    row[f'ZN mean (bps)'] = round(df[zn_col].mean(), 2)
                    row[f'ZN std (bps)'] = round(df[zn_col].std(), 2)
                row['|surprise| mean'] = round(df['surprise_std'].abs().mean(), 2)
                row['|surprise| std'] = round(df['surprise_std'].abs().std(), 2)
                row['N'] = len(df)
                return row


            t5_rows = {}
            # FPU regimes — from epu_clean
            t5_rows['Non-stress'] = summary_stats(
                epu_clean[epu_clean['fiscal_stress'] == 0], 'ret_sym5', 'zn_sym5')
            t5_rows['Fiscal stress'] = summary_stats(
                epu_clean[epu_clean['fiscal_stress'] == 1], 'ret_sym5', 'zn_sym5')
            # Shutdown phases — from shut_clean
            t5_rows['Pre-window'] = summary_stats(
                shut_clean[shut_clean['pre_window'] == 1], 'ret_sym5',
                'zn_sym5' if HAS_ZN else None)
            t5_rows['Active shutdown'] = summary_stats(
                shut_clean[shut_clean['shutdown'] == 1], 'ret_sym5',
                'zn_sym5' if HAS_ZN else None)
            t5_rows['Post-window'] = summary_stats(
                shut_clean[shut_clean['post_window'] == 1], 'ret_sym5',
                'zn_sym5' if HAS_ZN else None)
            table5 = pd.DataFrame(t5_rows).T
            table5.index.name = 'Regime / Phase'
            table5.to_excel(writer, sheet_name='Table5 Summary Stats')

    print(f"\n  Saved: {output_path}")

# 13. OVERLAP CHECK
if RUN_OVERLAP_CHECK and RUN_SHUTDOWN_MODELS:
    print("\n" + "=" * 60)
    print("SECTION 13: OVERLAP CHECK")
    print("=" * 60)

    overlap = shut_clean[(shut_clean['pre_window']==1) & (shut_clean['post_window']==1)]
    print(f"\n  Obs in both pre and post window: {len(overlap)}")
    if len(overlap) > 0:
        print(overlap['type'].value_counts().to_string())

    shut_excl = shut_clean[
        ~((shut_clean['pre_window']==1) & (shut_clean['post_window']==1))].copy()
    m11_excl = run_regression(shut_excl,
        'ret_sym5_abs ~ surprise_abs + shutdown + pre_window + post_window '
        '+ surprise_abs:shutdown + surprise_abs:pre_window + surprise_abs:post_window',
        'M11_excl')
    print(f"\n  M11 excl overlap (N={len(shut_excl)}):")
    for param in ['surprise_abs:shutdown','surprise_abs:pre_window','surprise_abs:post_window']:
        if param in m11_excl.params:
            p = m11_excl.pvalues[param]
            print(f"    {param}: {m11_excl.params[param]:.4f} bps {stars(p)} (p={p:.4f})")