from pathlib import Path
import sys,json,pickle
import numpy as np,pandas as pd
from scipy.stats import wilcoxon,pearsonr
# Writes figure source data to figures/source_data/. Run from any directory.
REPO=Path(__file__).resolve().parents[2];D=REPO/'figures/source_data';D.mkdir(exist_ok=True)
sys.path.insert(0,str(REPO/'notebooks'));import afidutils as au
model=REPO/'resources/models/afids_stn_model_all_data.pkl';test=pd.read_csv(REPO/'data/coordinate/test_data.csv').subjid.tolist()
def read(p):return pd.read_csv(p,comment='#',header=None).iloc[:,1:4].to_numpy(float)
def gt(ds,s):
 g=s.replace('sub-R','sub-P') if ds=='SNSXPD1.5T' else s.replace('sub-','sub-R') if ds=='3T7T_3T' else s
 fs=list((REPO/'data/coordinate/STN').glob(f'*/stn_mcp/{g}/anat/*groundtruth*.fcsv'));assert len(fs)==1
 v=read(fs[0]);v[1,0]*=-1;return g,v
rows=[]
for ds in ['SNSXPD','SNSXPD1.5T','3T7T_3T','3T7T_7T']:
 for file in sorted((REPO/f'data/coordinate/AFIDs/{ds}').glob('afids_mcp/*/anat/*groundtruth*.fcsv')):
  s=file.parts[-3]
  if s not in test:continue
  subj,t=gt(ds,s);p=au.predict_stn_coordinates(str(file),str(model))
  for h in range(2):
   r=dict(subject=subj,hemisphere=['R','L'][h],dataset=ds,method='AFIDs',ED=float(np.linalg.norm(p[h]-t[h])))
   r.update({prefix+k:float(v) for prefix,arr in [('pred_',p[h]),('truth_',t[h]),('err_',p[h]-t[h])] for k,v in zip('xyz',arr)});rows.append(r)
for ds in ['SNSXPD','SNSXPD1.5T']:
 for file in sorted((REPO/f'resources/LeadDBS/{ds}').glob('stn_mcp/*/anat/*.fcsv')):
  subj,t=gt(ds,file.parts[-3]);p=read(file);p[1,0]*=-1
  for h in range(2):
   r=dict(subject=subj,hemisphere=['R','L'][h],dataset=ds,method='Registration',ED=float(np.linalg.norm(p[h]-t[h])))
   r.update({prefix+k:float(v) for prefix,arr in [('pred_',p[h]),('truth_',t[h]),('err_',p[h]-t[h])] for k,v in zip('xyz',arr)});rows.append(r)
df=pd.DataFrame(rows);assert len(df)==120;assert not df.duplicated(['subject','hemisphere','dataset','method']).any()
# Regression guard: the regenerated predictions must match the committed ones.
if (D/'predictions.csv').exists():
 old=pd.read_csv(D/'predictions.csv');merged=df.merge(old,on=['subject','hemisphere','dataset','method'],suffixes=('_new','_old'));assert len(merged)==120;assert np.allclose(merged.ED_new,merged.ED_old)
df.to_csv(D/'predictions.csv',index=False)
base=REPO/'resources/MNI152NLin2009bAsym';files=sorted(base.glob('afids_mcp/*/anat/*desc-rater*.fcsv'));t=read(next(base.glob('stn_mcp/*/anat/*groundtruth*.fcsv')));t[1,0]*=-1;ag=read(next(base.glob('afids_mcp/*/anat/*groundtruth*.fcsv')));ids=np.array([1,2,3,4,5,6,7,8,9,11,12,13,15,16,17,18])-1
rows=[];afs=[]
for file in files:
 name=file.stem.split('desc-')[1].split('_')[0];coords=read(file);p=au.predict_stn_coordinates(str(file),str(model));err=np.linalg.norm(coords[ids]-ag[ids],axis=1).mean()
 for h in range(2):
  r=dict(annotation=name,hemisphere=['R','L'][h],AFID_error=err,ED=float(np.linalg.norm(p[h]-t[h])));r.update({prefix+k:float(v) for prefix,arr in [('pred_',p[h]),('truth_',t[h]),('err_',p[h]-t[h])] for k,v in zip('xyz',arr)});rows.append(r)
 for i in ids:afs.append(dict(annotation=name,AFID=int(i+1),x=coords[i,0],y=coords[i,1],z=coords[i,2]))
pd.DataFrame(rows).to_csv(D/'annotation_predictions.csv',index=False);pd.DataFrame(afs).to_csv(D/'annotation_coordinates.csv',index=False)
stats={};secondary=[]
for target in ['cZI','fct','RN']:
 v=pd.read_csv(REPO/f'data/{target}_logocv_results_unified.csv');summ=v.groupby('model').ed.agg(['median','mean','std']).sort_values('median');winner=summ.index[0];best=v[v.model==winner].copy();best['target']=target;secondary.append(best)
 assert best.groupby('fold').hemisphere.nunique().eq(2).all()
 stats[target]={'model':winner,'folds':best.fold.nunique(),'n':len(best),'median':best.ed.median(),'q25':best.ed.quantile(.25),'q75':best.ed.quantile(.75)}
pd.concat(secondary).to_csv(D/'secondary_targets.csv',index=False)
for ds in df.dataset.unique():
 v=df[(df.dataset==ds)&(df.method=='AFIDs')];stats[ds]={'n':len(v),'median':v.ED.median(),'q25':v.ED.quantile(.25),'q75':v.ED.quantile(.75),'under_2mm':float((v.ED<2).mean())}
json.dump(stats,open(D/'verified_summary.json','w'),indent=2);print(json.dumps(stats,indent=2))

raw=pd.read_csv(REPO/'data/coordinate/AFIDs/df_afids_mcp_raw.csv',index_col=0)
stn=pd.read_csv(REPO/'data/coordinate/STN/df_stn_mcp_raw.csv',index_col=0)
subs=[s for s in stn.index if s not in test and s in raw.index]
ar=au.remove_outliers(raw.loc[subs],raw.columns,7)
stn.loc[ar.index].to_csv(D/'development_stn.csv')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
clean=au.remove_outliers(raw,raw.columns,7)
X=au.flip_and_concatenate_hemispheres(clean).drop(index=test,errors='ignore')
X.to_csv(D/'fig2_features.csv')
pc=PCA().fit(StandardScaler().fit_transform(X));nonzero=X.columns[X.var()>1e-12]
X[nonzero].corr().to_csv(D/'fig2_correlations.csv')
pd.DataFrame({'pc':np.arange(1,34),'cumulative':pc.explained_variance_ratio_.cumsum(),'variance':pc.explained_variance_ratio_}).to_csv(D/'fig2_pca.csv',index=False)
w=(abs(pc.components_)*pc.explained_variance_ratio_[:,None]).sum(axis=0)
pd.Series(w,index=X.columns).groupby(lambda c:c[:-1]).sum().to_csv(D/'fig2_weights.csv',header=['weight'])
xr=au.flip_and_concatenate_hemispheres(ar).iloc[:len(ar)];t=stn.loc[xr.index,['RSTNx','RSTNy','RSTNz']]
pd.DataFrame({axis:{col:pearsonr(xr[col],t[tc]).statistic for col in nonzero} for axis,tc in zip('xyz',t.columns)}).to_csv(D/'fig2_stn_correlations.csv')
assert len(X)==320 and X.index.nunique()==160 and len(ar)==47

# --------------------------------------------------------------------------
# Secondary-target (cZI/fct/RN) ground-truth centroids, MCP-aligned.
# Same recovery method as notebooks/2.6_Generalized_Target_Prediction_unified.py
# (docstring there: verified to reproduce local afids_mcp fcsv coordinates to
# ~1e-5 mm by re-applying the recovered per-subject transform). Re-derived
# here (not copied from a cache) directly from thrower19/all_target_model_data
# (vendored at data/secondary_targets_thrower19/) + the local ITK transform .txt files.
THROWER = REPO/'data/secondary_targets_thrower19'
assert THROWER.exists(), f'missing {THROWER}'

def parse_itk_affine_txt(path):
 params=None
 with open(path) as f:
  for line in f:
   if line.startswith('Parameters:'):params=np.array([float(x) for x in line.split(':')[1].split()])
 rot=params[:9].reshape(3,3);trans=params[9:12];lps=np.eye(4);lps[:3,:3]=rot;lps[:3,3]=trans
 Dm=np.diag([-1.,-1.,1.,1.]);return np.linalg.inv(Dm@lps@Dm)

def subject_transform(sid):
 g=sorted((REPO/'data/coordinate/AFIDs/SNSXCN/afids_mcp'/sid/'anat').glob('*.txt'));assert len(g)==1;return parse_itk_affine_txt(g[0])

def raw_target_centroid(fcsv):
 d=pd.read_csv(fcsv,sep=',',header=2)
 r=d[d.label.str.lower().str.startswith('r_')][['x','y','z']].to_numpy()[0]
 l=d[d.label.str.lower().str.startswith('l_')][['x','y','z']].to_numpy()[0]
 return r,l

def apply_xform(m,xyz):
 h=np.append(xyz,1.);return (m@h)[:3]

sec_rows=[]
for target in ['cZI','fct','RN']:
 cdir=THROWER/'training_data/training_targets'/f'{target}_GT_centroids'
 ids=sorted({Path(f).name.split('_')[0] for f in cdir.glob('sub-*.fcsv')
             if (REPO/'data/coordinate/AFIDs/SNSXCN/afids_mcp'/Path(f).name.split('_')[0]).exists()})
 if target=='RN':assert 'sub-C008' in ids;ids.remove('sub-C008')
 for sid in ids:
  r,l=raw_target_centroid(cdir/f'{sid}_{target}_contacts.fcsv');m=subject_transform(sid)
  r_mcp=apply_xform(m,r);l_mcp=apply_xform(m,l);l_mcp[0]*=-1
  sec_rows.append(dict(target=target,subject=sid,hemisphere='R',x=r_mcp[0],y=r_mcp[1],z=r_mcp[2]))
  sec_rows.append(dict(target=target,subject=sid,hemisphere='L',x=l_mcp[0],y=l_mcp[1],z=l_mcp[2]))
sec_df=pd.DataFrame(sec_rows)
assert dict(sec_df.groupby('target').size())=={'cZI':64,'fct':64,'RN':62}
# Cross-check: re-applying the recovered transform to each subject's own raw
# AFIDs reproduces the local afids_mcp fcsv to <1e-3 mm (same check 2.6 already
# verified for these subjects/transforms; repeated here as a live guard).
chk_sid=sec_df.subject.iloc[0]
raw_afid_glob=sorted((REPO/'data/coordinate/AFIDs/SNSXCN/afids_groundtruth'/chk_sid/'anat').glob('*.fcsv'))
mcp_afid_glob=sorted((REPO/'data/coordinate/AFIDs/SNSXCN/afids_mcp'/chk_sid/'anat').glob('*.fcsv'))
assert raw_afid_glob and mcp_afid_glob
raw_afid=pd.read_csv(raw_afid_glob[0],sep=',',header=2)[['x','y','z']].to_numpy()
mcp_afid=pd.read_csv(mcp_afid_glob[0],sep=',',header=2)[['x','y','z']].to_numpy()
m=subject_transform(chk_sid)
reproj=np.array([apply_xform(m,row) for row in raw_afid])
assert np.allclose(reproj,mcp_afid,atol=1e-3),'transform-recovery guard failed'
sec_df.to_csv(D/'secondary_target_coords.csv',index=False)
print('secondary_target_coords.csv:',dict(sec_df.groupby('target').size()))
