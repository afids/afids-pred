from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import wilcoxon,pearsonr
from PIL import Image
REPO=Path(__file__).resolve().parents[2];OUT=REPO/'figures/manuscript';D=REPO/'figures/source_data'
v=pd.read_csv(D/'predictions.csv');assert len(v)==120
keys=['subject','hemisphere','dataset','method'];assert not v.duplicated(keys).any()
assert np.allclose(v.ED,np.linalg.norm(v[['err_x','err_y','err_z']],axis=1))
for k in 'xyz':assert np.allclose(v['pred_'+k]-v['truth_'+k],v['err_'+k])
audit={}
for ds,expected in [('SNSXPD1.5T',.000850677490234375),('SNSXPD',.5958194732666016)]:
 d=v[v.dataset==ds].pivot(index=['subject','hemisphere'],columns='method',values='ED');assert len(d)==20;p=wilcoxon(d.Registration-d.AFIDs).pvalue;assert np.isclose(p,expected);audit[ds+'_benchmark_p']=p
for d1,d2 in [('SNSXPD1.5T','SNSXPD'),('3T7T_3T','3T7T_7T')]:
 d=v[(v.method=='AFIDs')&v.dataset.isin([d1,d2])].pivot(index=['subject','hemisphere'],columns='dataset',values='ED');assert len(d)==20;audit[d1+'_acquisition_p']=wilcoxon(np.round(d[d1]-d[d2],3)).pvalue
ann=pd.read_csv(D/'annotation_predictions.csv');assert len(ann)==76;assert ann.annotation.nunique()==38;audit['annotation_r']=pearsonr(ann.AFID_error,ann.ED).statistic
assert len(pd.read_csv(D/'development_stn.csv'))==47
sec=pd.read_csv(D/'secondary_targets.csv')
for t,n in [('cZI',64),('fct',64),('RN',62)]:
 s=sec[sec.target==t];assert len(s)==n;assert s.groupby('fold').hemisphere.nunique().eq(2).all();assert np.allclose(s.ed,np.linalg.norm(s[['err_x','err_y','err_z']],axis=1))
meta={}
# Every rendered figure.
for p in sorted(OUT.glob('Figure_*.png')):
 name=p.stem.split('_',1)[1]
 if name=='5':continue
 with Image.open(p) as im:
  im.load();dpi=im.info['dpi'];assert min(dpi)>=600;assert abs(im.width/dpi[0]*25.4-180)<.1;meta[name]={'pixels':im.size,'dpi':dpi,'width_mm':im.width/dpi[0]*25.4,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
json.dump({'numerical_checks':audit,'exports':meta},open(D/'Verification.json','w'),indent=2);print(f'Verified {len(meta)} exports, keyed prediction pairing, residual norms, annotation sample sizes, and secondary-target fold counts.')
