"""Comprehensive leaderboard for FedShield decision-making.

Reports per-dataset, per-attack, per-(dataset,attack), and per-cell rankings
across ASR, Acc, and combined defense_score. Counts wins, ties, and losses.
"""
from pathlib import Path
import pandas as pd
import re

KNOWN_ATTACKS = [
    'label_flip+backdoor+sign_flip+scaling','sign_flip+scaling','backdoor+scaling',
    'sign_flip','label_flip','backdoor','scaling','noise_update','clean'
]
ATTACK_ALT = '|'.join(re.escape(a) for a in KNOWN_ATTACKS)
RE = re.compile(
    rf'^proto_(?P<dataset>[^_]+)_(?P<variant>.+?)_(?P<attack>{ATTACK_ALT})_'
    rf'r(?P<mr>\d{{2}})_s(?P<seed>\d{{2}})(?:_n(?P<n>\d{{2}}))?_metrics\.csv$'
)
V = {
    'fedavg':'FedAvg','krum':'Krum','multi_krum':'M-Krum','trimmed_mean':'Trim',
    'fltrust':'FLTr','median':'CMed','fedshield_v10':'FS','fedshield_v10_a025':'FS-PD'
}
D = {'mitbih':'MIT-BIH','ciciomt':'CIC-IoMT','ptbxl':'PTB-XL','physionet2017':'P2017'}
ATK_DISP = {
    'sign_flip':'sign','scaling':'scale','label_flip':'lblflip','backdoor':'backdr',
    'noise_update':'noise','sign_flip+scaling':'sign+sc','backdoor+scaling':'bd+sc',
    'label_flip+backdoor+sign_flip+scaling':'fullcomp'
}
ATK_ORDER = ['sign_flip','scaling','label_flip','backdoor','noise_update',
             'sign_flip+scaling','backdoor+scaling',
             'label_flip+backdoor+sign_flip+scaling']
OUR = ['fedshield_v10','fedshield_v10_a025']


def load_df():
    rows = []
    for csv in Path('results/proto').rglob('proto_*_metrics.csv'):
        m = RE.match(csv.name)
        if not m or m.group('n'): continue
        if m.group('variant') not in V or m.group('dataset') not in D: continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if len(df) == 0: continue
        last = df.iloc[-1]
        acc = float(last.get('acc', float('nan')))
        asr = float(last.get('asr', float('nan')))
        rows.append({
            'ds':   m.group('dataset'),
            'v':    m.group('variant'),
            'atk':  m.group('attack'),
            'mr':   round(float(m.group('mr'))/100, 2),
            'seed': int(m.group('seed')),
            'acc':  acc,
            'asr':  asr,
            'sc':   acc * (1 - asr),
        })
    return pd.DataFrame(rows)


def section(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


def lb1_headline(df):
    section('LEADERBOARD 1: HEADLINE PER DATASET (32 attacked cells x 5 seeds)')
    for metric, lower in [('asr', True), ('acc', False), ('sc', False)]:
        dirn = 'lower=better' if lower else 'higher=better'
        print(f'\n--- {metric.upper()} ({dirn}) ---')
        ag = df.groupby(['ds', 'v'])[metric].mean().unstack('v')
        for ds in D:
            s = ag.loc[ds].sort_values(ascending=lower)
            line = f'  {D[ds]:9s}: '
            for rank, (v, val) in enumerate(s.items(), 1):
                mark = '*' if v in OUR else ' '
                line += f'{rank}.{V[v]}({val:.3f}){mark}  '
            print(line)


def lb2_per_attack_ranks(df):
    section('LEADERBOARD 2: FS / FS-PD RANKS PER (DATASET, ATTACK), avg over rho')
    for metric, lower in [('asr', True), ('sc', False)]:
        dirn = 'lower=better' if lower else 'higher=better'
        print(f'\n--- ranking by {metric.upper()} ({dirn}) ---')
        cols = [ATK_DISP[a] for a in ATK_ORDER]
        header = '  ' + 'dataset+ '.ljust(13) + ' '.join(f'{c:>8s}' for c in cols)
        print(header)
        for ds in D:
            sub = df[df.ds == ds].groupby(['atk', 'v'])[metric].mean().unstack('v')
            line_fs = f'  {D[ds][:9]:9s} FS  '
            line_pd = f'  {D[ds][:9]:9s} PD  '
            for atk in ATK_ORDER:
                if atk not in sub.index:
                    line_fs += '       --'
                    line_pd += '       --'
                    continue
                row = sub.loc[atk]
                sorted_ = row.sort_values(ascending=lower)
                ranks = {v: i + 1 for i, v in enumerate(sorted_.index)}
                line_fs += f'{ranks["fedshield_v10"]:>8d} '
                line_pd += f'{ranks["fedshield_v10_a025"]:>8d} '
            print(line_fs)
            print(line_pd)


def lb3_win_counts(df):
    section('LEADERBOARD 3: WIN/TIE COUNTS')
    print('\n--- Per-(dataset, attack), avg over rho [32 cells] ---')
    for metric, lower in [('asr', True), ('sc', False)]:
        rows = []
        for ds in D:
            sub = df[df.ds == ds].groupby(['atk', 'v'])[metric].mean().unstack('v')
            for atk in sub.index:
                row = sub.loc[atk]
                best = row.min() if lower else row.max()
                for v in row.index:
                    is_winner = (row[v] == best)
                    near = abs(row[v] - best) < 0.005
                    rows.append({'v': v, 'win': is_winner, 'near': near})
        cdf = pd.DataFrame(rows)
        print(f'  {metric.upper()}:')
        for v in OUR:
            sub = cdf[cdf.v == v]
            print(f'    {V[v]:6s}  wins={int(sub["win"].sum()):2d}/32   '
                  f'within_0.005={int(sub["near"].sum()):2d}/32')

    print('\n--- Per-(dataset, attack, rho) full cells [128 cells] ---')
    for metric, lower in [('asr', True), ('sc', False)]:
        rows = []
        for ds in D:
            sub = df[df.ds == ds].groupby(['atk', 'mr', 'v'])[metric].mean().unstack('v')
            for idx in sub.index:
                row = sub.loc[idx]
                best = row.min() if lower else row.max()
                for v in row.index:
                    is_winner = (row[v] == best)
                    near = abs(row[v] - best) < 0.005
                    rows.append({'v': v, 'win': is_winner, 'near': near})
        cdf = pd.DataFrame(rows)
        print(f'  {metric.upper()}:')
        for v in OUR:
            sub = cdf[cdf.v == v]
            tot = len(sub)
            print(f'    {V[v]:6s}  wins={int(sub["win"].sum()):3d}/{tot}'
                  f' = {100*sub["win"].sum()/tot:.1f}%   '
                  f'within_0.005={int(sub["near"].sum()):3d}/{tot}'
                  f' = {100*sub["near"].sum()/tot:.1f}%')


def lb4_by_attack(df):
    section('LEADERBOARD 4: BY ATTACK FAMILY (avg across datasets, ratios, seeds)')
    for metric, lower in [('asr', True), ('sc', False)]:
        dirn = 'lower=better' if lower else 'higher=better'
        print(f'\n--- {metric.upper()} ({dirn}) ---')
        ag = df.groupby(['atk', 'v'])[metric].mean().unstack('v')
        print(f'  {"attack":9s} | rank-1 method (val)   | FS rank (val)  | FS-PD rank (val)')
        for atk in ATK_ORDER:
            if atk not in ag.index: continue
            row = ag.loc[atk]
            sorted_ = row.sort_values(ascending=lower)
            winner = sorted_.index[0]
            w_val = sorted_.iloc[0]
            fs_rank = list(sorted_.index).index('fedshield_v10') + 1
            pd_rank = list(sorted_.index).index('fedshield_v10_a025') + 1
            fs_val = row['fedshield_v10']
            pd_val = row['fedshield_v10_a025']
            print(f'  {ATK_DISP[atk]:9s} | {V[winner]:5s} ({w_val:.3f})       '
                  f'| FS={fs_rank} ({fs_val:.3f}) | PD={pd_rank} ({pd_val:.3f})')


def lb5_high_adversary(df):
    section('LEADERBOARD 5: HIGH-ADVERSARY (rho_m=0.4) STRESS TEST')
    for metric, lower in [('asr', True), ('sc', False)]:
        dirn = 'lower=better' if lower else 'higher=better'
        print(f'\n--- {metric.upper()} at rho_m=0.4 ({dirn}) ---')
        sub = df[df.mr == 0.4].groupby(['ds', 'v'])[metric].mean().unstack('v')
        for ds in D:
            s = sub.loc[ds].sort_values(ascending=lower)
            line = f'  {D[ds]:9s}: '
            for rank, (v, val) in enumerate(s.items(), 1):
                mark = '*' if v in OUR else ' '
                line += f'{rank}.{V[v]}({val:.3f}){mark} '
            print(line)


def lb6_summary_decision(df):
    section('LEADERBOARD 6: DECISION SUMMARY (where do we win, where do we lose)')
    print('\nFor each (dataset, metric), what rank does FS / FS-PD achieve?')
    print('Lower rank is better.')
    for metric, lower in [('asr', True), ('acc', False), ('sc', False)]:
        print(f'\n--- {metric.upper()} ---')
        ag = df.groupby(['ds', 'v'])[metric].mean().unstack('v')
        print(f'  {"dataset":9s}  FS rank   FS-PD rank   winner')
        for ds in D:
            s = ag.loc[ds].sort_values(ascending=lower)
            ranks = {v: i + 1 for i, v in enumerate(s.index)}
            winner = s.index[0]
            print(f'  {D[ds]:9s}  {ranks["fedshield_v10"]:^7d}  '
                  f'{ranks["fedshield_v10_a025"]:^10d}   {V[winner]}({s.iloc[0]:.3f})')


def main():
    df = load_df()
    df = df[df.mr > 0].reset_index(drop=True)
    print(f'Loaded {len(df)} runs across {df.ds.nunique()} datasets, '
          f'{df.v.nunique()} methods, {df.atk.nunique()} attacks, '
          f'{df.mr.nunique()} ratios, {df.seed.nunique()} seeds.')
    lb1_headline(df)
    lb2_per_attack_ranks(df)
    lb3_win_counts(df)
    lb4_by_attack(df)
    lb5_high_adversary(df)
    lb6_summary_decision(df)


if __name__ == '__main__':
    main()
