"""Read-only end-to-end consistency check of final TC OPE and inference outputs.
Run: python scripts/08_verify_final_inference.py
No outputs are created or modified. Does not train or evaluate policies.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'outputs' / 'results' / 'retention'
SUMMARY = RESULTS / 'formal_retention_summary_tc.csv'
ORIGINS = RESULTS / 'formal_retention_origin_level_ope_tc.csv'
STATS = RESULTS / 'formal_retention_vs_c28_statistics.csv'
SENS = ROOT / 'outputs' / 'results' / 'sensitivity' / 'eval_gamma_095'
CONDITIONS = ['C1','C2','C3','C5','C10','C16','C19','C28']
KEY = ['Condition','Algorithm','Evaluator','Seed']
EXPECTED = pd.MultiIndex.from_product([CONDITIONS,['CQL','IQL'],['ARX','XGBoost'],range(1,21)],names=KEY)

def check(cond, message):
    if not bool(cond):
        raise SystemExit('FAIL: '+message)

def full_keys(df, name):
    check(set(KEY).issubset(df.columns),name+' missing key columns')
    k = pd.MultiIndex.from_frame(df[KEY])
    check(k.is_unique,name+' duplicate keys')
    check(len(df)==640 and len(k.difference(EXPECTED))==0 and len(EXPECTED.difference(k))==0,name+' key coverage')

def close(a,b,name,tol=1e-9):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    check(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),name+' nonfinite or shape mismatch')
    err=float(np.max(np.abs(a-b))) if a.size else 0.0
    check(err<=tol,f'{name} max error {err:.9g} > {tol}')
    print(f'PASS {name}: max abs error={err:.3g}')

print('Read-only final inference verification')
s=pd.read_csv(SUMMARY)
t=pd.read_csv(STATS)
has_origins = ORIGINS.is_file()
o = pd.read_csv(ORIGINS) if has_origins else None
if not has_origins:
    print('SKIP private origin-level checks: origin-level OPE file not distributed')

full_keys(s,'summary')
if has_origins:
    check(len(o)==688000,'origin rows != 688000')
    check(set(KEY+['Fragment','Discounted_Return','Mean_Reward']).issubset(o.columns),'missing origin columns')
    check(not o.duplicated(KEY+['Fragment','Origin_Time']).any(),'duplicate origin identifiers')

check(len(t)==28,'statistics row count')
check(not t.duplicated(['Condition','Algorithm','Evaluator']).any(),'duplicate statistical comparisons')
check(set(t.Condition)==set(CONDITIONS[:-1]),'comparison conditions')
check((t.Reference=='C28').all(),'wrong reference')

if has_origins:
    # Independently reproduce the official equal-weight-over-fragments aggregation.
    frag=o.groupby(KEY+['Fragment'],sort=False).agg(
        Discounted_Return=('Discounted_Return','mean'),Mean_Reward=('Mean_Reward','mean'),
        N_Origins=('Discounted_Return','size')).reset_index()
    agg=frag.groupby(KEY,sort=False).agg(
        Mean_Discounted_Return=('Discounted_Return','mean'),
        Mean_Reward=('Mean_Reward','mean'),
        N_Evaluation_Fragments=('Fragment','nunique'),
        N_Origins=('N_Origins','sum')).reset_index()
    m=s.merge(agg,on=KEY,validate='one_to_one',suffixes=('_summary','_origin'))
    check(len(m)==640,'aggregation join incomplete')
    for col in ['Mean_Discounted_Return','Mean_Reward','N_Evaluation_Fragments','N_Origins']:
        close(m[col+'_summary'],m[col+'_origin'],'origin-to-summary '+col)
    check((m.N_Evaluation_Fragments_summary==28).all(),'not all policies have 28 fragments')
    print('PASS origin-level aggregation (640 policies, 28 evaluation fragments)')


# Check that policy-level statistics really derive from the current summary.
rows=[]
for algorithm in ['CQL','IQL']:
    for evaluator in ['ARX','XGBoost']:
        subset=s[(s.Algorithm==algorithm)&(s.Evaluator==evaluator)]
        ref=subset[subset.Condition=='C28'].set_index('Seed').Mean_Discounted_Return
        fam=[]
        for c in CONDITIONS[:-1]:
            cur=subset[subset.Condition==c].set_index('Seed').Mean_Discounted_Return
            check(set(cur.index)==set(ref.index)==set(range(1,21)),'unmatched seeds')
            d=(cur-ref).sort_index().to_numpy(dtype=float)
            p=float(wilcoxon(d,alternative='two-sided',zero_method='wilcox',method='auto').pvalue)
            fam.append((c,d,p))
        order=np.argsort([r[2] for r in fam]); p=np.array([r[2] for r in fam]); adj=np.empty(7)
        adj[order]=np.maximum.accumulate(np.minimum(1.,(7-np.arange(7))*p[order]))
        for i,(c,d,pval) in enumerate(fam):
            rows.append(dict(Algorithm=algorithm,Evaluator=evaluator,Condition=c,
                Mean_Difference_Ck_minus_C28=float(d.mean()),
                Median_Difference_Ck_minus_C28=float(np.median(d)),
                Wilcoxon_P_Raw=pval,Holm_P_Adjusted=float(adj[i]),
                Cohen_dz=float(d.mean()/d.std(ddof=1)) if d.std(ddof=1)>0 else np.nan))
r=pd.DataFrame(rows)
z=t.merge(r,on=['Algorithm','Evaluator','Condition'],validate='one_to_one',suffixes=('_published','_independent'))
check(len(z)==28,'statistic join incomplete')
for col in ['Mean_Difference_Ck_minus_C28','Median_Difference_Ck_minus_C28','Wilcoxon_P_Raw','Holm_P_Adjusted','Cohen_dz']:
    close(z[col+'_published'],z[col+'_independent'],'independent statistics '+col,1e-9)

prov=STATS.with_suffix('.provenance.json')
check(prov.exists(),'missing primary provenance')
p=json.loads(prov.read_text(encoding='utf-8'))
check(p.get('source_sha256')==hashlib.sha256(SUMMARY.read_bytes()).hexdigest(),'statistics source hash stale')
check(p.get('statistics_sha256')==hashlib.sha256(STATS.read_bytes()).hexdigest(),'statistics hash stale')
print('PASS primary statistical provenance')

# Gamma discount: verify policy-level sensitivity against origin-level values independently.
policy_path=SENS/'gamma_095_policy_values.csv'
check(policy_path.exists(),'missing policy_values.csv')
g=pd.read_csv(policy_path)
full_keys(g,'gamma095 policies')
check((s.Horizon_Hours==2).all(),'not a 2-step horizon')
check(np.isfinite(g.Mean_Discounted_Return_Gamma095).all(),'nonfinite gamma095 values')
if has_origins:
    # Independent algebra using the raw origin-level mean reward and discounted return,
    # not the already-aggregated policy summary.
    # For each origin: G_95 = G_99 + (.95-.99)/(1-.99) * (2*mean_reward-G_99)
    origin_gamma=o.assign(Gamma095=o.Discounted_Return + (-4.0)*(2*o.Mean_Reward-o.Discounted_Return))
    f095=origin_gamma.groupby(KEY+['Fragment'],sort=False).Gamma095.mean().reset_index()
    ind=f095.groupby(KEY,sort=False).Gamma095.mean().reset_index()
    join=g.merge(ind,on=KEY,validate='one_to_one')
    check(len(join)==640,'gamma095 join incomplete')
    close(join.Mean_Discounted_Return_Gamma095,join.Gamma095,'gamma095 origin-to-policy',1e-8)
    # Cross-check 2-step consistency using physically expected rewards (reward=-(energy+underheat)).
    if {'Mean_Energy_Component','Mean_Underheat_Component'}.issubset(o.columns):
        close(o.Mean_Reward, -(o.Mean_Energy_Component+o.Mean_Underheat_Component),
              'origin reward components',1e-8)

# Public-only check: two-step discount conversion commutes with
# equal-weight aggregation across origins and fragments.
public_join = g[KEY + ['Mean_Discounted_Return_Gamma095']].merge(
    s[KEY + ['Mean_Discounted_Return', 'Mean_Reward']],
    on=KEY, validate='one_to_one'
)
check(len(public_join)==640, 'public gamma095 join incomplete')
expected_gamma095 = (
    public_join.Mean_Discounted_Return
    - 4.0 * (
        2.0 * public_join.Mean_Reward
        - public_join.Mean_Discounted_Return
    )
)
close(
    public_join.Mean_Discounted_Return_Gamma095,
    expected_gamma095,
    'gamma095 policy-level discount conversion',
    1e-8
)

# Recompute gamma095 matched-seed differences, Wilcoxon and Holm against saved sensitivity stats.
st095=pd.read_csv(SENS/'gamma_095_retention_vs_c28_statistics.csv')
check(len(st095)==28,'gamma095 statistical row count')
for a in ['CQL','IQL']:
    for e in ['ARX','XGBoost']:
        sub=g[(g.Algorithm==a)&(g.Evaluator==e)]
        ref=sub[sub.Condition=='C28'].set_index('Seed').Mean_Discounted_Return_Gamma095
        pp=[]; dd=[]
        for c in CONDITIONS[:-1]:
            cur=sub[sub.Condition==c].set_index('Seed').Mean_Discounted_Return_Gamma095
            d=(cur-ref).sort_index().to_numpy();dd.append(d)
            pp.append(float(wilcoxon(d).pvalue))
        order=np.argsort(pp); adjusted=np.empty(7)
        adjusted[order]=np.maximum.accumulate(np.minimum(1.,(7-np.arange(7))*np.array(pp)[order]))
        for i,c in enumerate(CONDITIONS[:-1]):
            row=st095[(st095.Algorithm==a)&(st095.Evaluator==e)&(st095.Condition==c)]
            check(len(row)==1,'missing gamma095 comparison')
            close([row.Mean_Difference_Ck_minus_C28.iloc[0],row.Wilcoxon_P_Raw.iloc[0],row.Holm_P_Adjusted.iloc[0]],
                  [dd[i].mean(),pp[i],adjusted[i]],f'gamma095 stats {a}/{e}/{c}',1e-9)
print('PASS gamma095 independent statistical comparisons')
if has_origins:
    print('ALL CHECKS PASSED (including private origin-level verification).')
else:
    print('PUBLIC CHECKS PASSED; private origin-level checks SKIPPED.')
print('No files were modified by verification.')
