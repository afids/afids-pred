from pathlib import Path
import sys,json,os
ROOT=Path(__file__).resolve().parents[2]
from drawing import *
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from scipy.stats import wilcoxon
from scipy.ndimage import distance_transform_edt
from statsmodels.stats.multitest import multipletests

SEG=EXT/'segmentations_binary_MNIspace'
def seg_mask(hemi,target):
 return nib.load(SEG/f'SNSX32NLin2020Asym_space-MNI152NLin2009bAsym_hemi-{hemi}_desc-{target}_mask.nii.gz')
FLASH_PATH=EXT/'Synthesized_FLASH25_in_MNI_v2_200um.nii.gz'
OUT=ROOT/'figures/manuscript';base.OUT=OUT;NEW=ROOT/'figures/source_data';BASE=NEW
class Canvas(base.Canvas):
 def save(self,name):
  self.f.savefig(OUT/f'{name}.png',dpi=610)
  if os.environ.get('AFIDS_VECTOR'):  # editable copies for manual figure work (gitignored)
   vec=ROOT/'figures/panels/full';vec.mkdir(parents=True,exist_ok=True)
   for ext in ('pdf','svg'):self.f.savefig(vec/f'{name}.{ext}',dpi=610)
  PREVIEW.mkdir(exist_ok=True)
  with Image.open(OUT/f'{name}.png') as im:
   im.resize((1260,round(1260*self.h/180))).save(PREVIEW/f'{name}.png')
   print(name,im.size,flush=True)
  plt.close(self.f)

plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,'xtick.labelsize':6.2,'ytick.labelsize':6.2})
X=pd.read_csv(BASE/'fig2_features.csv',index_col=0);sc=StandardScaler().fit(X);pca=PCA().fit(sc.transform(X));scores=pca.transform(sc.transform(X));names=['AC','PC','ICS','PMJ','SIPF','SLMS','ILMS','IMS','MB','LVAC','LVPC']
# Desaturated viridis: keeps the full hue journey (so the 11 landmarks stay
# distinguishable) but blended toward mid-grey, so it reads as restrained
# rather than a loud, fully-saturated rainbow -- toward grey rather than
# white so points still hold contrast against the white page.
_raw=plt.cm.viridis(np.linspace(.05,.92,11));_grey=np.array([.58,.58,.58,1])
colors=[c*.55+_grey*.45 for c in _raw]
col=dict(zip(names,colors))
geom=np.load(OLD/'reference_review/geometry.npz');p=geom['native'];mcp=(p[0]+p[1])/2;ey=p[0]-p[1];ey/=np.linalg.norm(ey);ex=np.cross(p[3]-mcp,ey);ex/=np.linalg.norm(ex);ez=np.cross(ex,ey);rot=np.array([ex,ey,ez])
def native(v):return np.asarray(v)@rot+mcp
DS=['SNSXPD1.5T','SNSXPD','3T7T_3T','3T7T_7T'];DC=[CORAL,TEAL,PURPLE,'#546C92'];DL=['1.5 T + gad','7 T','3 T','7 T'];data=pd.read_csv(NEW/'predictions.csv')

def frame3(a,lims,aspect=None,view=(16,35),zoom=1.2):
 a.view_init(elev=view[0],azim=view[1]);a.set(xlim=lims[0],ylim=lims[1],zlim=lims[2]);a.set_box_aspect(aspect or [b-a for a,b in lims],zoom=zoom);a.set_proj_type('ortho');a.axis('off')

def landmark_cloud(C,pos,labels=True):
 im,vol=anatomy();a=C.ax(*pos,projection='3d',computed_zorder=False);plane(a,im,vol,'sag',[-63,25,-35,42],0,res=180);means=[]
 for name in names:
  v=native(X[[name+k for k in 'xyz']].values);mu=v.mean(0);means.append(mu);a.scatter(*v.T,s=1.35,c=[col[name]],alpha=.20,depthshade=False,zorder=4);a.scatter(*mu,s=23,c=[col[name]],edgecolors='white',linewidth=.7,depthshade=False,zorder=6)
 surfaces(a,names=('STN',));frame3(a,[(-7,40),(-65,25),(-35,42)],zoom=1.27)
 if labels:
  C.f.canvas.draw();pts=[]
  for mu,name in zip(means,names):xp,yp,_=proj3d.proj_transform(*mu,a.get_proj());pts.append((xp,yp,name))
  pts.sort(key=lambda k:-k[1])
  for (xp,yp,name),yy in zip(pts,np.linspace(.94,.04,11)):
   a.annotate(name,(xp,yp),xytext=(.96,yy),textcoords='axes fraction',ha='left',va='center',fontsize=6,color=INK,arrowprops={'arrowstyle':'-','lw':.45,'color':col[name],'shrinkB':3},zorder=10)
 return a

def atlasmesh(a,alpha=.4,hemi=None):
 for h in [hemi] if hemi else ['lh','rh']:
  im=nib.load(OLD/'atlas_sources'/f'{h}_STN.nii.gz');v=im.get_fdata();verts,faces,_,_=marching_cubes(gaussian_filter(v,.65),.5,step_size=2);verts=nib.affines.apply_affine(im.affine,verts)
  a.add_collection3d(Poly3DCollection(verts[faces],facecolors='#AAB7C1',edgecolor='none',alpha=alpha,shade=True,lightsource=LightSource(azdeg=300,altdeg=50),zorder=2))

def ecdf(a,v,c,label=None,lw=1.2):
 v=np.sort(np.asarray(v));a.step(np.r_[0,v],np.r_[0,np.arange(1,len(v)+1)/len(v)],where='post',color=c,lw=lw,label=label)

TARGET_COLORS={'STN':CORAL,'cZI':PURPLE,'fct':'#CA9528','RN':TEAL}  # matches Figure 4's tcols

RATER_COLORS=('#4C6FA5','#C9932B','#6FA5A0')

def venn_majority(ax,colors=RATER_COLORS):
 # Purely abstract (no anatomical image, no specific-structure shape): 3
 # raters as overlapping disks, majority (>=2 of 3) region shaded, its
 # centroid marked as ground truth. Reverted from a real-STN-shape version
 # per feedback -- the clean abstraction reads better here.
 r=1.0;offs=[(-.55,.30),(.55,.30),(0,-.62)]
 g=np.linspace(-2,2,220);GX,GY=np.meshgrid(g,g);count=np.zeros_like(GX)
 for (ox,oy),c in zip(offs,colors):
  ax.add_patch(Circle((ox,oy),r,facecolor=c,edgecolor=c,alpha=.22,lw=1.1,zorder=2))
  ax.add_patch(Circle((ox,oy),r,facecolor='none',edgecolor=c,lw=1.1,zorder=4))
  count+=((GX-ox)**2+(GY-oy)**2<=r**2)
 ax.contourf(GX,GY,count,levels=[1.5,3.5],colors=[CORAL],alpha=.6,zorder=3)
 mask=count>=2;comx,comy=GX[mask].mean(),GY[mask].mean()
 ax.scatter(comx,comy,s=24,c=CORAL,edgecolor='white',lw=.7,zorder=6)
 ax.set_xlim(-1.85,1.85);ax.set_ylim(-1.95,1.75);ax.set_aspect('equal');ax.axis('off')

def fig1(out_name='Figure_1'):
 # Compact landscape relayout, per reference mockup: venn (top-left) overlaps
 # the lens (bottom-left); the 3D cloud sits to the lens's right, connected to
 # it by a plain black tangent-line callout (no bezel/shadow/vignette -- a
 # single black rim reads as a magnifier without extra chrome); the learning
 # pipeline (matrices, fold bar, per-target winners) runs down the right
 # column. No panel letters/captions baked in -- annotations are added
 # separately on the right, per request.
 C=Canvas(84)
 sec=pd.read_csv(NEW/'secondary_target_coords.csv')
 stn=pd.read_csv(NEW/'development_stn.csv',index_col=0);target_allv={}
 for target in TARGET_COLORS:
  allv=[]
  for h in ['R','L']:
   if target=='STN':
    v=stn[[h+'STN'+k for k in 'xyz']].to_numpy()
    if h=='L':v[:,0]*=-1
   else:
    v=sec[(sec.target==target)&(sec.hemisphere==h)][['x','y','z']].to_numpy()
   allv.append(v)
  target_allv[target]=np.concatenate(allv)

 # 3D coordinate cloud. Each landmark's dispersion is a set of translucent
 # filled 3D disks (Poly3DCollection) rather than thin wireframe rings, which
 # can visually self-intersect/squiggle at an oblique angle.
 a=C.ax(58,2,50,78,projection='3d',computed_zorder=False)
 means=[]
 for name in names:
  v=X[[name+k for k in 'xyz']].to_numpy();mu=v.mean(0);means.append(mu)
  a.scatter(v[:,0],v[:,1],np.full(len(v),-28),s=1,c=MUTE,alpha=.05,depthshade=False,zorder=1)
  a.scatter(*v.T,s=3.0,c=MUTE,alpha=.28,depthshade=False,zorder=5)
  w,V=np.linalg.eigh(np.cov(v.T));w=np.maximum(w,0);tt=np.linspace(0,2*np.pi,60)
  for j,k in [(0,1),(1,2),(0,2)]:
   ring=mu+np.cos(tt)[:,None]*np.sqrt(w[j])*V[:,j]+np.sin(tt)[:,None]*np.sqrt(w[k])*V[:,k]
   poly=Poly3DCollection([ring],facecolor=MUTE,edgecolor=MUTE,lw=.4,alpha=.14,zorder=2)
   a.add_collection3d(poly)
  # dashed drop-line to the floor -- an explicit depth cue so the plot reads
  # unambiguously as 3D rather than a scatter of flat blobs.
  a.plot([mu[0],mu[0]],[mu[1],mu[1]],[mu[2],-28],c=LIGHT,lw=.6,ls=(0,(1.5,1.5)),zorder=1)
  a.scatter(*mu,s=28,c=INK,edgecolor='white',lw=.7,depthshade=False,zorder=8)
 for xx in [0,10,20,30]:a.plot([xx,xx],[-37,23],[-28,-28],c=LIGHT,lw=.4,zorder=0)
 for yy in [-30,-20,-10,0,10,20]:a.plot([0,33],[yy,yy],[-28,-28],c=LIGHT,lw=.4,zorder=0)
 frame3(a,[(-3,34),(-38,24),(-29,40)],view=(21,38),zoom=1.35)
 for target,c in TARGET_COLORS.items():
  for v,m in zip(np.split(target_allv[target],2),['o','s']):
   a.scatter(*v.T,s=7,c=c,marker=m,alpha=.8,edgecolor='white',lw=.18,depthshade=False,zorder=9)

 C.f.canvas.draw()
 def pagepoint3(v):
  xp,yp,_=proj3d.proj_transform(*v,a.get_proj())
  return C.a.transData.inverted().transform(a.transData.transform((xp,yp)))
 allpage=np.concatenate([np.array([pagepoint3(v) for v in target_allv[t]]) for t in TARGET_COLORS])
 center=allpage.mean(0);radius=max(np.linalg.norm(allpage-center,axis=1))+.6
 C.a.add_patch(Circle(center,radius,fill=False,ec='black',lw=1.0))

 # Lens -- a plain black-rimmed magnifier (no bezel/shadow/vignette/scale
 # bar): real 200-um 7-T anatomy, all four targets' own segmentation
 # contours, and their ground-truth points. Sits to the left of the cloud;
 # a black tangent-line callout (below) links it to the circled cluster.
 flash_im=nib.load(FLASH_PATH);flash_vol=np.asanyarray(flash_im.dataobj,dtype='float32')
 z=float(geom['targets_native'][:,2].mean());cx,cy=9,-17;mag_r=20
 xx,yy=np.meshgrid(np.linspace(cx-mag_r,cx+mag_r,640),np.linspace(cy-mag_r,cy+mag_r,640));zz=np.full_like(xx,z);vv=sample(flash_im,flash_vol,xx,yy,zz)
 b=C.ax(4,17,50,50);b.set_xlim(cx-mag_r,cx+mag_r);b.set_ylim(cy-mag_r,cy+mag_r)
 clip=Circle((cx,cy),mag_r,transform=b.transData)
 im=b.imshow(vv,origin='lower',extent=(cx-mag_r,cx+mag_r,cy-mag_r,cy+mag_r),cmap='gray',vmin=np.percentile(vv,1),vmax=np.percentile(vv,99.4));im.set_clip_path(clip);b.axis('off')
 for target,c in [('STN',CORAL),('RN','#CBD3D8'),('cZI',PURPLE),('fct','#CA9528')]:
  ai=seg_mask('R',target);av=sample(ai,ai.get_fdata(),xx,yy,zz);cont=b.contour(xx,yy,av,levels=[.5],colors=[c],linewidths=1.3);cont.set_clip_path(clip)
 for target,c in TARGET_COLORS.items():
  for v,m in zip(np.split(target_allv[target],2),['o','s']):
   nv=native(v);scat=b.scatter(nv[:,0],nv[:,1],s=9,c=c,marker=m,alpha=.75,edgecolor='white',lw=.25,zorder=5);scat.set_clip_path(clip)
 b.add_patch(Circle((cx,cy),mag_r,fill=False,ec='black',lw=2.2,zorder=7))
 def bpoint(ang_deg):
  ang=np.radians(ang_deg)
  return C.a.transData.inverted().transform(b.transData.transform((cx+mag_r*np.cos(ang),cy+mag_r*np.sin(ang))))
 t1=bpoint(-25);t2=bpoint(25)
 C.line(center[0]-radius*.7,center[1]-radius*.5,*t2,c='black',lw=1.0)
 C.line(center[0]-radius*.7,center[1]+radius*.5,*t1,c='black',lw=1.0)

 # Ground-truth-annotation reference (abstract, no anatomical image): three
 # raters' segmentations overlap; majority (>=2 of 3) is ground truth.
 # Overlaps the lens's top-left corner, with an explicit rater/overlap key.
 vax=C.ax(4,2,20,17);venn_majority(vax)
 raters=[('Rater 1',RATER_COLORS[1]),('Rater 2',RATER_COLORS[2]),('Rater 3',RATER_COLORS[0]),('Overlap',CORAL)]
 for i,(lab,c) in enumerate(raters):
  yy=4+i*3.4
  C.a.add_patch(Rectangle((26,yy-1.2),3,2.4,fc=c,ec='none'))
  C.t(30,yy,lab,5.8,c=INK,style='italic')

 # Learning pipeline, right column -- coordinates -> scale+PCA -> per-target
 # model screen -> selected architecture. No single model is named as "the"
 # model, since the winner genuinely differs by target.
 h=C.ax(112,4,22,30);h.imshow(sc.transform(X)[:32],cmap=DIV,vmin=-2.5,vmax=2.5,aspect='equal');h.axis('off')
 C.t(123,1.8,'33 coordinates',5.8,c=MUTE,ha='center',style='italic')
 h=C.ax(142,4,28,30);h.imshow(scores[:32,:23],cmap='PuOr',vmin=-5,vmax=5,aspect='equal');h.axis('off')
 C.t(156,1.8,'Scale + PCA · 23 PCs',5.8,c=MUTE,ha='center',style='italic')

 # Compact model-selection widget: a small cluster of dots stands in for the
 # 10-candidate architecture search space (previously unshown), feeding into
 # a nested-fold diagram -- an outer 4-fold bar (dark = held-out) with a
 # bracket dropping to a smaller 3-block inner row (hyperparameter tuning),
 # so "nested" is actually visible rather than asserted in a caption.
 C.a.add_patch(Rectangle((112,36.6),15,7.6,fc='#F4F6F7',ec='none',zorder=10))
 for i in range(10):
  C.a.scatter([114.2+2.4*(i%5)],[39 if i<5 else 42],s=5,c=MUTE,alpha=.85,zorder=12)
 C.t(119.5,45.3,'10-model search space',5.2,c=MUTE,ha='center',style='italic')
 C.arrow(128,40.4,132,40.4,c=MUTE)
 ow=3.2
 for i in range(4):
  C.a.add_patch(Rectangle((133+i*4.0,37.5),ow,3.2,fc=INK if i==0 else '#E2E5E8',ec='none'))
 iw=2.2
 for i in range(3):
  C.a.add_patch(Rectangle((138.8+i*2.7,42.3),iw,1.8,fc='#BFC7CD' if i==1 else '#DEE2E5',ec='none'))
 C.line(137,40.7,138.8,42.3,c=MUTE,lw=.5);C.line(148.2,40.7,146.4,42.3,c=MUTE,lw=.5)
 C.t(140.6,45.3,'4 outer x 3 inner folds',5.2,c=MUTE,ha='center',style='italic')

 # STN's number here is its real HELD-OUT median (Figure 3; AFIDs/Ridge
 # predictions, pooled across all 4 acquisitions x 20 hemispheres = 80) --
 # not the internal nested-CV screening median (0.85 mm, Supplementary
 # Figure S3). cZI/fct/RN have no held-out test, so their numbers below are
 # necessarily their internal-CV medians instead.
 stn_med=float(data[data.method=='AFIDs'].ED.median())
 stats=json.load(open(NEW/'verified_summary.json'))
 # basis differs by row: STN has a real held-out test (Figure 3), cZI/fct/RN
 # don't, so theirs are internal-CV medians (Figure 4d) instead -- called out
 # once here rather than implying all four are on the same footing.
 winners=[('STN',CORAL,'Ridge',stn_med),('cZI',TARGET_COLORS['cZI'],stats['cZI']['model'],stats['cZI']['median']),
          ('fct',TARGET_COLORS['fct'],stats['fct']['model'],stats['fct']['median']),
          ('RN',TARGET_COLORS['RN'],stats['RN']['model'],stats['RN']['median'])]
 C.t(145,49,'Selected architecture per target',5.8,c=MUTE,ha='center',style='italic')
 C.t(145,52.3,'Median error – STN: held-out (Fig. 3); others: internal CV (Fig. 4c–d)',4.8,c=MUTE,ha='center',style='italic')
 cx0=112
 for i,(t,c,model,med) in enumerate(winners):
  yy=57+i*6.4
  C.a.add_patch(Rectangle((cx0,yy-2.6),66,5.2,fc='#F4F6F7',ec='none'))
  C.a.add_patch(Rectangle((cx0,yy-2.6),1.3,5.2,fc=c,ec='none'))
  C.a.scatter([cx0+5],[yy],s=16,c=c);C.t(cx0+7.5,yy,t,6.6,bold=True)
  C.t(cx0+19,yy,model,6.6,c=INK)
  C.t(cx0+48,yy,f'median {med:.2f} mm',5.9,c=MUTE)
 C.save(out_name)
 return C.h

def fig2():
 C=Canvas(195);corr=pd.read_csv(BASE/'fig2_correlations.csv',index_col=0);tc=pd.read_csv(BASE/'fig2_stn_correlations.csv',index_col=0)
 C.panel(4,7,'a','Coordinate coupling');C.panel(109,7,'b','Population in PCA space')
 order=[f for name in names for f in corr.index if f[:-1]==name];lab=[f[:-1]+' '+f[-1] for f in order];bounds=[i for i in range(1,len(order)) if order[i][:-1]!=order[i-1][:-1]]
 a=C.ax(17,18,71,71);h=a.imshow(corr.loc[order,order],cmap=DIV,vmin=-1,vmax=1);a.set_xticks(range(28));a.set_xticklabels(lab,rotation=90,fontsize=5.7);a.set_yticks(range(28));a.set_yticklabels(lab,fontsize=5.7);a.tick_params(length=0,pad=1)
 for sp in a.spines.values():sp.set_visible(False)
 for k in bounds:a.axhline(k-.5,c='white',lw=.8);a.axvline(k-.5,c='white',lw=.8)
 a=C.ax(91,18,9,71);a.imshow(tc.loc[order],cmap=DIV,vmin=-1,vmax=1,aspect='auto');a.set(xticks=[0,1,2],xticklabels=['x','y','z'],yticks=[]);a.set_title('STN',fontsize=6,pad=3);a.tick_params(length=0,pad=1)
 for sp in a.spines.values():sp.set_visible(False)
 cb=C.f.colorbar(h,cax=C.ax(26,105,50,2),orientation='horizontal');cb.set_ticks([-1,0,1]);cb.outline.set_visible(False);cb.ax.tick_params(length=2,labelsize=6,pad=1);C.t(81,106,'r',7)
 # Oblique 3D scatter: a light, minimal box (not the default heavy grey grid)
 # plus a floor-projected shadow for extra depth. Unlike Figure 1's 11
 # well-separated cluster means, this is one diffuse ~320-point population --
 # a fully frameless floor-shadow (no box, no drop-lines) tried first read as
 # two disconnected blobs rather than one 3D cloud, so a light box frame is
 # kept here as the spatial reference the shadow alone couldn't provide.
 a=C.ax(112,17,54,53,projection='3d',computed_zorder=False)
 zlo=-8
 a.scatter(scores[:,0],scores[:,1],np.full(len(scores),zlo),s=2,c=LIGHT,alpha=.25,depthshade=False,zorder=1)
 a.scatter(scores[:,0],scores[:,1],scores[:,2],s=10,c=scores[:,0],cmap='Purples',alpha=.8,edgecolor='white',linewidth=.15,depthshade=False,zorder=5)
 a.set(xlim=(-8,6),ylim=(-8,6),zlim=(zlo,6))
 a.view_init(23,-57);a.set_proj_type('ortho');a.set_box_aspect([1,1,1],zoom=1.25)
 a.set_xlabel(f'PC1 · {pca.explained_variance_ratio_[0]:.0%}',labelpad=-3,fontsize=6.3,color=MUTE)
 a.set_ylabel(f'PC2 · {pca.explained_variance_ratio_[1]:.0%}',labelpad=-3,fontsize=6.3,color=MUTE)
 a.set_zlabel(f'PC3 · {pca.explained_variance_ratio_[2]:.0%}',labelpad=-5,fontsize=6.3,color=MUTE)
 a.tick_params(labelsize=5,pad=-2,colors=MUTE);a.grid(True,color=LIGHT,linewidth=.4)
 for axis in [a.xaxis,a.yaxis,a.zaxis]:axis.pane.fill=False;axis.pane.set_edgecolor(LIGHT);axis.line.set_color(LIGHT)
 a=C.ax(121,85,50,22);cum=pca.explained_variance_ratio_.cumsum()
 a.fill_between(range(1,34),cum*100,color=PURPLE,alpha=.12,lw=0)
 a.plot(range(1,34),cum*100,c=PURPLE,lw=1.3);a.scatter(23,cum[22]*100,c=CORAL,s=23,zorder=5);a.axhline(99,c=LIGHT,lw=.55,ls='--');a.text(11,38,'23 PCs / 99%',fontsize=7,color=PURPLE);a.set(xlim=(1,33),ylim=(0,105),xticks=[1,16,23,33],yticks=[0,50,99],xlabel='Components',ylabel='Variance (%)');tidy(a)
 C.panel(4,126,'c','Principal modes in anatomical space');C.panel(112,126,'d','PCA contributions')
 # Real midline-sagittal 200-um anatomy (same FLASH25 template as Figure 1)
 # behind the mode shifts, instead of blank axes -- "anatomical space" is the
 # panel's own title, so the mode lines now sit on the anatomy they describe.
 flash_im=nib.load(FLASH_PATH);flash_vol=np.asanyarray(flash_im.dataobj,dtype='float32')
 gy,gz=np.meshgrid(np.linspace(-34,20,360),np.linspace(-25,38,360))
 npts=native(np.column_stack([np.zeros(gy.size),gy.ravel(),gz.ravel()]))
 sag=sample(flash_im,flash_vol,npts[:,0].reshape(gy.shape),npts[:,1].reshape(gy.shape),npts[:,2].reshape(gy.shape))
 for mode in range(2):
  a=C.ax(13+mode*49,140,38,34)
  a.imshow(sag,origin='lower',extent=(-34,20,-25,38),cmap='gray',vmin=np.percentile(sag,1),vmax=np.percentile(sag,99),zorder=0)
  dire=np.zeros((2,33));dire[:,mode]=np.array([-2,2])*np.sqrt(pca.explained_variance_[mode]);vals=sc.inverse_transform(pca.inverse_transform(dire))
  for name in names:
   ids=[X.columns.get_loc(name+k) for k in 'xyz'];v=vals[:,ids];a.plot(v[:,1],v[:,2],c=col[name],lw=1.4,zorder=3,path_effects=[pe.withStroke(linewidth=2.4,foreground='white')])
   a.scatter(v[0,1],v[0,2],s=16,facecolors='white',edgecolors=[col[name]],linewidth=.8,zorder=4);a.scatter(v[1,1],v[1,2],s=16,c=[col[name]],linewidth=0,zorder=4)
  a.set_title('PC'+str(mode+1),fontsize=7,pad=4,c='white',bbox=dict(fc=INK,ec='none',pad=2))
  a.set(xlim=(-34,20),ylim=(-25,38),xticks=[-20,0,20],yticks=[-20,0,20],xlabel='y (mm)',ylabel='z (mm)' if mode==0 else '');tidy(a)
 # Landmark color key -- Figure 1 now identifies AFIDs by label rather than
 # hue, but panel c still needs per-landmark color to read at all, so the
 # mapping is spelled out here instead of assumed. Placed below the mode
 # plots' own tick labels/axis titles (which extend past their nominal
 # 174 mm box bottom), not on top of them.
 for i,name in enumerate(names):
  x0=13+i*7.9
  C.a.add_patch(Rectangle((x0,187),2,2,fc=col[name],ec='none'));C.t(x0+2.7,188,name,5.2,c=INK)
 C.t(13,193,'Open: −2 SD; filled: +2 SD',6.1,c=MUTE)
 weights=(abs(pca.components_)*pca.explained_variance_ratio_[:,None]).sum(0);wt=pd.Series(weights,index=X.columns);arr=np.array([[wt[name+k] for k in 'xyz'] for name in names]);a=C.ax(128,136,24,42);im=a.imshow(arr,cmap=SEQ,aspect='auto',vmin=0,vmax=arr.max());a.set(xticks=[0,1,2],xticklabels=['x','y','z'],yticks=range(11),yticklabels=names);a.tick_params(length=0,labelsize=6)
 for sp in a.spines.values():sp.set_visible(False)
 cb=C.f.colorbar(im,cax=C.ax(157,136,2,42));cb.outline.set_visible(False);cb.ax.tick_params(length=2,labelsize=5.8)
 C.save('Figure_2')

def residual_axis(a,d,c,title,lim=2.5):
 a.add_patch(Circle((0,0),1,fc='#F7F8F9',ec='#B5BEC6',lw=.65,ls='--',zorder=0));a.axhline(0,c=LIGHT,lw=.5);a.axvline(0,c=LIGHT,lw=.5)
 for _,r in d.iterrows():a.plot([0,r.err_y],[0,r.err_z],c=c,alpha=.15,lw=.5,zorder=1)
 a.scatter(d.err_y,d.err_z,s=15,c=c,edgecolors='white',linewidth=.35,zorder=3);a.scatter(0,0,marker='+',s=40,c=INK,lw=.8,zorder=4);a.set(aspect='equal',xlim=(-lim,lim),ylim=(-lim,lim),xticks=[-2,0,2],yticks=[-2,0,2]);a.set_title(title,fontsize=7,pad=5);tidy(a)

def fig3():
 C=Canvas(177);C.panel(4,7,'a','Where predictions fall');C.t(175,7,'Reference centered · y–z projection',6.3,c=MUTE,ha='right')
 for j,(ds,c,lab) in enumerate(zip(DS,DC,DL)):
  a=C.ax(13+j*42,18,32,34);residual_axis(a,data[(data.dataset==ds)&(data.method=='AFIDs')],c,lab);a.set_xlabel('Δy (mm)',fontsize=6.3)
  if j==0:a.set_ylabel('Δz (mm)',fontsize=6.3)
 C.t(48,62,'Clinical pair',6.5,c=MUTE,ha='center');C.t(132,62,'External pair',6.5,c=MUTE,ha='center')
 C.panel(4,74,'b','Full error distributions');C.panel(103,74,'c','Within-participant contrast changes')
 a=C.ax(14,86,73,36)
 for ds,c,lab in zip(DS,DC,['Clinical 1.5 T + gad','Clinical 7 T','External 3 T','External 7 T']):ecdf(a,data[(data.dataset==ds)&(data.method=='AFIDs')].ED,c,lab)
 a.set(xlim=(0,2.3),ylim=(0,1.03),xticks=[0,1,2],yticks=[0,.5,1],xlabel='STN error (mm)',ylabel='Fraction');tidy(a);a.legend(loc='lower right',fontsize=5.5,frameon=False,handlelength=1.2,labelspacing=.2)
 a=C.ax(119,86,50,36)
 for j,(d1,d2) in enumerate([(DS[0],DS[1]),(DS[2],DS[3])]):
  pp=data[(data.method=='AFIDs')&data.dataset.isin([d1,d2])].pivot(index=['subject','hemisphere'],columns='dataset',values='ED');v=pp[d1]-pp[d2];cloud(a,v,j,DC[j*2]);pv=wilcoxon(np.round(v,3)).pvalue;a.text(j,1.25,f'p = {pv:.2f}',ha='center',fontsize=6)
 a.axhline(0,c=INK,lw=.65,ls='--');a.set(xlim=(-.6,1.6),ylim=(-1.4,1.5),yticks=[-1,0,1],xticks=[0,1],xticklabels=['1.5 T − 7 T','3 T − 7 T'],ylabel='Δ error (mm)');tidy(a)
 C.panel(4,139,'d','Registration benchmark');C.panel(103,139,'e','Paired gain')
 C.t(31,146,'1.5 T + gad',6,c=MUTE,ha='center');C.t(69,146,'7 T',6,c=MUTE,ha='center')
 a=C.ax(14,149,73,21)
 for j,ds in enumerate(DS[:2]):
  dd=data[data.dataset==ds].pivot(index=['subject','hemisphere'],columns='method',values='ED');c=DC[j];xs=[j*2.4,j*2.4+1]
  for _,r in dd.iterrows():a.plot(xs,[r.Registration,r.AFIDs],c=LIGHT,lw=.5);a.scatter(xs,[r.Registration,r.AFIDs],s=8,c=['#8D9AA4',c],edgecolor='white',lw=.2,zorder=3)
  for xx,v,cc in [(xs[0],dd.Registration,'#8D9AA4'),(xs[1],dd.AFIDs,c)]:lo,med,hi=np.percentile(v,[25,50,75]);a.plot([xx,xx],[lo,hi],c=cc,lw=3);a.scatter(xx,med,s=15,c='white',edgecolor=cc,zorder=5)
 a.set(xticks=[0,1,2.4,3.4],xticklabels=['Atlas','AFIDs','Atlas','AFIDs'],ylim=(0,3.5),yticks=[0,1.5,3],ylabel='Error (mm)');tidy(a)
 a=C.ax(119,149,50,21)
 for j,ds in enumerate(DS[:2]):
  dd=data[data.dataset==ds].pivot(index=['subject','hemisphere'],columns='method',values='ED');v=dd.Registration-dd.AFIDs;cloud(a,v,j,DC[j]);pv=wilcoxon(v).pvalue;a.text(j,2.33,'p = '+('0.00085' if j==0 else '0.60'),fontsize=5.8,ha='center')
 a.axhline(0,c=INK,ls='--',lw=.6);a.set(xticks=[0,1],xticklabels=['1.5 T + gad','7 T'],ylim=(-1,2.8),yticks=[-1,0,1,2],ylabel='Atlas − AFIDs (mm)');tidy(a)
 C.save('Figure_3')

def fig4():
 C=Canvas(177);ann=pd.read_csv(NEW/'annotation_predictions.csv');secondary=pd.read_csv(NEW/'secondary_targets.csv')
 C.panel(4,7,'a','Annotation-to-target variability');C.panel(108,7,'b','Error propagation')
 # Real 3D STN surface model (bilateral, DISTAL atlas) with actual
 # prediction points scattered around it, in native template space -- now on
 # a real 200-um FLASH25 axial plane (same template as Figure 1/2) instead
 # of the standard MNI152 hires T2 coronal plane.
 a=C.ax(-3,14,100,72,projection='3d',computed_zorder=False)
 hemi_pts={}
 for h in ['R','L']:
  dd=ann[ann.hemisphere==h];v=dd[['pred_'+k for k in 'xyz']].to_numpy();gt=dd[['truth_'+k for k in 'xyz']].iloc[0].to_numpy()
  if h=='L':v[:,0]*=-1;gt[0]*=-1
  hemi_pts[h]=(native(v),native(gt))
 zfix=float(np.mean([hemi_pts['R'][1][2],hemi_pts['L'][1][2]]))
 flash_im=nib.load(FLASH_PATH);flash_vol=np.asanyarray(flash_im.dataobj,dtype='float32')
 plane(a,flash_im,flash_vol,'ax',[-23,23,-25,3],zfix,res=200);atlasmesh(a,.28)
 for h,(v,gt) in hemi_pts.items():
  for pt in v:a.plot([gt[0],pt[0]],[gt[1],pt[1]],[gt[2],pt[2]],c=CORAL,alpha=.22,lw=.5,zorder=3)
  a.scatter(*v.T,s=13,c=CORAL,alpha=.8,edgecolor='white',lw=.3,depthshade=False,zorder=6);a.scatter(*gt,s=48,marker='*',c=INK,edgecolor='white',lw=.6,depthshade=False,zorder=7)
 frame3(a,[(-23,23),(-25,3),(-22,4)],view=(90,-90),zoom=1.55)
 C.t(9,90,'38 annotation sets · 76 hemisphere predictions',6.3,c=MUTE);C.a.scatter(10,96,s=26,marker='*',c=INK);C.t(13,96,'Reference',6.2,c=MUTE);C.a.scatter(36,96,s=13,c=CORAL);C.t(39,96,'Prediction',6.2,c=MUTE);C.a.scatter(62,96,s=30,marker='s',c='#AAB7C1',alpha=.7,edgecolor='none');C.t(65,96,'STN atlas',6.2,c=MUTE)
 a=C.ax(120,20,51,55);xx=np.linspace(ann.AFID_error.min(),ann.AFID_error.max(),100);fit=sm.OLS(ann.ED,sm.add_constant(ann.AFID_error)).fit();pr=fit.get_prediction(sm.add_constant(xx)).summary_frame();a.fill_between(xx,pr.mean_ci_lower,pr.mean_ci_upper,color='#D7ECEC',zorder=0)
 for h,m in [('R','o'),('L','s')]:dd=ann[ann.hemisphere==h];a.scatter(dd.AFID_error,dd.ED,s=15,c=TEAL,marker=m,alpha=.75,edgecolor='white',lw=.3,zorder=3)
 a.plot(xx,pr['mean'],c=INK,lw=1);a.set(xlabel='AFID error (mm)',ylabel='STN error (mm)',xticks=[.5,1,1.5,2],yticks=[.5,1,1.5,2,2.5]);tidy(a);a.text(.05,.96,'r = 0.47\nSlope = 0.46',transform=a.transAxes,va='top',fontsize=7)
 C.t(120,85,'Circles: R; squares: L',6.2,c=MUTE)
 C.panel(4,105,'c','Beyond the STN');C.panel(108,105,'d','Architecture screen')
 C.t(10,113,'Internal cross-validation',6.3,c=MUTE)
 a=C.ax(14,122,70,42);tcols={'cZI':PURPLE,'fct':'#CA9528','RN':TEAL}
 for target,c in tcols.items():v=secondary[secondary.target==target].ed;ecdf(a,v,c,f'{target}  {np.median(v):.2f} mm')
 a.set(xlim=(0,max(secondary.ed)*1.02),ylim=(0,1.03),xticks=[0,1,2,3],yticks=[0,.5,1],xlabel='Target error (mm)',ylabel='Fraction');tidy(a);a.legend(loc='lower right',fontsize=6,frameon=False,handlelength=1.3,title='Median',title_fontsize=6)
 models=['Ridge','BayesianRidge','ElasticNet','Lasso','PLS','RandomForest','XGBoost','SVR','KNN','MLP'];arr=np.empty((10,3))
 for j,target in enumerate(['cZI','fct','RN']):arr[:,j]=pd.read_csv(REPO/f'data/{target}_logocv_results_unified.csv').groupby('model').ed.median().reindex(models)
 a=C.ax(137,116,23,49);h=a.imshow(arr,cmap='magma_r',aspect='auto',vmin=0,vmax=3);a.set(xticks=[0,1,2],xticklabels=['cZI','fct','RN'],yticks=range(10),yticklabels=models);a.tick_params(length=0,labelsize=6)
 for sp in a.spines.values():sp.set_visible(False)
 for j in range(3):i=np.argmin(arr[:,j]);a.add_patch(Rectangle((j-.47,i-.47),.94,.94,fill=False,ec=TEAL,lw=1.1))
 cb=C.f.colorbar(h,cax=C.ax(166,116,2,49));cb.set_label('Median error (mm)',fontsize=6,labelpad=3);cb.outline.set_visible(False);cb.ax.tick_params(labelsize=5.8,length=2)
 C.t(10,175,'cZI/fct: 32 participants; RN: 31. Model selection and error estimation use the same CV.',6.2,c=MUTE)
 C.save('Figure_4')

# Not part of the manuscript's own S1-S6 scheme (no paragraph anywhere cites
# it by name) -- kept available but deliberately unnumbered rather than
# dropped outright, in case it's wanted again later.
def supp_axis_errors_unused():
 C=Canvas(98);C.panel(4,7,'a','Clinical pair');C.panel(94,7,'b','External pair')
 for group,(d1,d2) in enumerate([(DS[0],DS[1]),(DS[2],DS[3])]):
  dd=data[(data.method=='AFIDs')&data.dataset.isin([d1,d2])].copy();dd['group']=dd.dataset
  for j,axis in enumerate('xyz'):
   dd['square']=dd['err_'+axis]**2;a=C.ax(12+group*88+j*26,20,19,58);pp=base.paired(a,dd,d1,d2,'square',[DC[group*2],DC[group*2+1]],['1.5 T\n+ gad' if group==0 else '3 T','7 T'],(0,4.2));a.set_title(axis,fontsize=8,pad=12);p=wilcoxon(np.round(pp[d1]-pp[d2],3)).pvalue;a.text(.5,1.03,f'p = {p:.2f}',transform=a.transAxes,ha='center',fontsize=6.4)
   if j==0:a.set_ylabel('Squared error (mm²)')
 C.t(5,94,'20 paired hemispheres per comparison. Lines preserve pairing; thick bars: IQR; ticks: median.',6.2,c=MUTE)
 C.save('Figure_S_axis_errors_unused')

SUPP=REPO/'data/supplementary'

def supp2():
 C=Canvas(98);h=pd.read_csv(SUPP/'stn_model_screen_per_hemisphere.csv');st=pd.read_csv(SUPP/'ridge_vs_xgb.csv').set_index('metric')
 dd=h[h.model.isin(['Ridge','XGBoost'])].groupby(['model','subject'])[['se_x','se_y','se_z','ed']].mean().reset_index()
 dd['group']=dd.model;dd['hemisphere']='mean'
 C.panel(-2,7,'','Ridge vs. XGBoost, paired per participant')
 for j,(axis,metric) in enumerate([('x','se_x'),('y','se_y'),('z','se_z'),('ED','ed')]):
  a=C.ax(11+j*44,20,30,58)
  base.paired(a,dd,'Ridge','XGBoost',metric,[PURPLE,'#8194A3'],['Ridge','XGBoost'],(0,dd[metric].max()*1.08))
  p=st.loc[metric,'p_bonf'];a.set_title(axis,fontsize=8,pad=12)
  a.text(.5,1.03,'p < 0.001' if p<1e-3 else f'p = {p:.2g}',transform=a.transAxes,ha='center',fontsize=6.4)
  if j==0:a.set_ylabel('Squared error (mm²)')
  if j==3:a.set_ylabel('Euclidean distance (mm)')
 C.t(5,90,'n = 50 training participants (hemispheres averaged); 5 × 4-fold grouped CV. Paired Wilcoxon, Bonferroni-corrected across 4 metrics.',6.2,c=MUTE)
 C.save('Figure_S3')

# Four-target architecture screen (author's v4 "combined" design, one column per
# target): ranked forest marks -- thin line = full range, thick bar = IQR, dot =
# median -- with the selected model in its target colour. Holm-corrected paired
# Wilcoxon p values vs. the selected model are recomputed here from the per-
# hemisphere data. STN pairs on participant means (n = 50; the two hemispheres of
# a participant are not independent), matching Table S2; cZI/fct/RN pair on
# hemisphere estimates, matching Tables S5-S7.
SCREENS=[('STN','Ridge','100 hemispheres · 50 participants','5 × 4-fold grouped CV'),
         ('cZI','ElasticNet','64 hemispheres · 32 participants','32-fold leave-one-participant-out'),
         ('fct','RandomForest','64 hemispheres · 32 participants','32-fold leave-one-participant-out'),
         ('RN','Lasso','62 hemispheres · 31 participants','31-fold leave-one-participant-out')]
def screen_data(target,winner):
 if target=='STN':
  h=pd.read_csv(SUPP/'stn_model_screen_per_hemisphere.csv');piv=h.pivot_table(index=['subject','hemisphere'],columns='model',values='ed')
  pair=h.groupby(['model','subject']).ed.mean().unstack(0)
 else:
  d=pd.read_csv(REPO/f'data/{target}_logocv_results_unified.csv');piv=d.pivot_table(index=['fold','hemisphere'],columns='model',values='ed');pair=piv
 order=piv.median().sort_values().index.tolist();others=[m for m in order if m!=winner]
 assert order[0]==winner,(target,order[0])
 p=[wilcoxon(pair[winner]-pair[m]).pvalue for m in others]
 return piv,order,dict(zip(others,multipletests(p,method='holm')[1]))
def supp3():
 C=Canvas(104);C.panel(-2,6,'','Architecture screen across the four targets')
 for j,(target,winner,n1,n2) in enumerate(SCREENS):
  piv,order,holm=screen_data(target,winner);x0=.5+j*44.3;tc=TARGET_COLORS[target]
  cap=float(np.ceil(piv.drop(columns='MLP').max().max()*1.05*2)/2)
  C.t(x0+17,15.5,target,8.4,bold=True,c=tc);C.t(x0+17,20,n1,5.0,c=MUTE);C.t(x0+17,23.4,n2,5.0,c=MUTE)
  C.t(x0+42.8,27.2,'Holm p',5.0,c=MUTE,ha='right')
  a=C.ax(x0+17,30,19,58)
  for i,m in enumerate(order):
   v=piv[m].dropna().to_numpy();lo,q1,med,q3,hi=np.percentile(v,[0,25,50,75,100]);w=m==winner;c=tc if w else '#8194A3'
   a.plot([lo,min(hi,cap)],[i,i],c=LIGHT,lw=.9,zorder=1,solid_capstyle='round')
   a.plot([q1,min(q3,cap)],[i,i],c=c,lw=2.6 if w else 1.6,zorder=2,solid_capstyle='round')
   a.scatter(min(med,cap),i,s=26 if w else 11,c=c,edgecolor='white',lw=.6,zorder=3)
   if hi>cap:
    a.annotate('',xy=(cap,i),xytext=(cap*.93,i),arrowprops=dict(arrowstyle='-|>',color=INK,lw=.6,shrinkA=0,shrinkB=0,mutation_scale=5),zorder=4)
    a.text(cap,i-.25,f'to {hi:.1f}',ha='right',va='bottom',fontsize=4.6,color=MUTE,style='italic')
   if w:a.text(med,i-.38,f'{med:.2f}',ha='center',va='bottom',fontsize=5.4,color=tc,weight='bold')
   pv=holm.get(m);a.text(1.36,i,'—' if pv is None else ('<0.001' if pv<1e-3 else f'{pv:.3f}'),transform=a.get_yaxis_transform(),
                        fontsize=5.0,color=INK if (pv is not None and pv<.05) else MUTE,ha='right',va='center')
  a.set(xlim=(0,cap*1.03),ylim=(len(order)-.5,-.8),yticks=range(len(order)),xlabel='Error (mm)')
  a.set_yticklabels(order,fontsize=5.6);a.tick_params(axis='y',length=0,pad=2);a.tick_params(axis='x',labelsize=5.6)
  a.spines['left'].set_visible(False);a.xaxis.grid(True,color=LIGHT,lw=.4);tidy(a)
 C.t(4,97,'Dot: median; thick bar: IQR; thin line: range (arrow: truncated, true maximum given). Holm p: paired Wilcoxon signed-rank vs. the selected model',5.6,c=MUTE)
 C.t(4,100.6,'(STN: participant means, n = 50; cZI/fct/RN: hemisphere estimates). Internal cross-validation only; STN held-out results are in Figure 3.',5.6,c=MUTE)
 C.save('Figure_S2')

def supp4():
 C=Canvas(70);pc=pd.read_csv(SUPP/'pca_threshold_sensitivity.csv');lc=pd.read_csv(SUPP/'learning_curve.csv')
 C.panel(4,6,'a','PCA variance threshold');C.panel(92,6,'b','Learning curve')
 a=C.ax(12,16,70,40);k=pc.n_components.to_numpy()
 a.plot(k,pc.test_median_ED,'-o',c=PURPLE,ms=3,lw=1,label='Held-out test');a.plot(k,pc.cv_median_ED,'-o',c='#8194A3',ms=3,lw=1,label='Training CV')
 for xi,yi,v in zip(k,pc.test_median_ED,pc.variance.astype(str)):
  a.annotate('none' if v.startswith('none') else f'{float(v)*100:g}%',(xi,yi),xytext=(0,-9) if v=='0.995' else (0,4),textcoords='offset points',ha='center',fontsize=5,color=MUTE)
 r=pc[pc.variance.astype(str)=='0.99'];a.scatter(r.n_components,r.test_median_ED,s=40,facecolor='none',edgecolor=CORAL,lw=.9,zorder=6)
 a.set(xlabel='Retained components',ylabel='Median ED (mm)',ylim=(0.7,1.45));a.legend(frameon=False,fontsize=5.6,loc='lower right');tidy(a)
 a=C.ax(100,16,70,40);g=lc.groupby('n_train_subjects').test_median_ED;n=g.median().index.to_numpy()
 a.fill_between(n,g.quantile(.1),g.quantile(.9),color=PURPLE,alpha=.15,lw=0);a.plot(n,g.median(),'-o',c=PURPLE,ms=3,lw=1)
 a.set(xlabel='Training participants',ylabel='Test median ED (mm)');tidy(a)
 C.t(5,66,'Test: 80 held-out STNs. CV: 5 × 4-fold grouped CV on 50 training participants. b: 25 random draws per size (line: median; band: 10th–90th percentile).',6.0,c=MUTE)
 C.save('Figure_S5')

# Leave-one-AFID-out reconstruction (notebooks/supplementary_analyses.py
# leave_one_afid_out): each landmark predicted from the other ten, 5 x 4-fold
# participant-grouped CV on the 50 training participants. Points are rows
# (hemisphere examples, mean over repeats).
def supp5():
 C=Canvas(81);r=pd.read_csv(SUPP/'leave_one_afid_out_per_row.csv');s=pd.read_csv(SUPP/'leave_one_afid_out.csv').set_index('landmark')
 per=r.groupby(['landmark','subject','hemisphere'])[['ed','ed_trainmean']].mean().reset_index()
 mid=[n for n in names if n not in ('LVAC','LVPC')];mid=sorted(mid,key=lambda n:s.loc[n,'median_ED']);order=mid+['LVAC','LVPC']
 C.panel(-2,6,'','Leave-one-AFID-out landmark reconstruction')
 a=C.ax(14,19,160,42);cap=5.0
 for i,n in enumerate(order):
  xp=i+(.6 if n in ('LVAC','LVPC') else 0);v=per[per.landmark==n].ed.clip(upper=cap)
  cloud(a,v,xp,col[n] if n in mid else '#A9B3BB')
  a.text(xp,1.04,f"{s.loc[n,'median_ED']:.2f}",transform=a.get_xaxis_transform(),ha='center',va='bottom',fontsize=5.6,color=INK)
 a.axvline(len(mid)-.2,c=LIGHT,lw=.6,ls='--')
 a.set(xlim=(-.6,len(order)+.2),ylim=(0,cap*1.04),xticks=[i+(.6 if n in ('LVAC','LVPC') else 0) for i,n in enumerate(order)],ylabel='Euclidean distance (mm)')
 a.set_xticklabels([n+('*' if n in ('AC','PC') else '') for n in order],fontsize=6.2);tidy(a)
 C.t(4,15.6,'Median (mm)',5.6,c=MUTE)
 p9,p7=s.loc['POOLED_9_midbrain'],s.loc['POOLED_7_midbrain_excl_AC_PC']
 C.t(4,69.5,f"Pooled median (IQR): nine midbrain landmarks {p9.median_ED:.2f} ({p9.q25_ED:.2f}–{p9.q75_ED:.2f}) mm; excluding AC/PC {p7.median_ED:.2f} ({p7.q25_ED:.2f}–{p7.q75_ED:.2f}) mm. LVAC/LVPC (ventricular, right of dashed line) are not pooled.",5.6,c=MUTE)
 C.t(4,73.5,'*AC and PC are exact mirror images in AC–PC/MCP space (ACy = −PCy; x = z = 0), so each is reconstructed trivially from the other.',5.6,c=MUTE)
 C.t(4,77.5,'Points: hemisphere examples of the 50 training participants (mean over 5 × 4-fold participant-grouped CV); errors above 5 mm drawn at 5 mm.',5.6,c=MUTE)
 C.save('Figure_S4')

def supp6():
 C=Canvas(70);ab=pd.read_csv(SUPP/'landmark_ablation.csv').sort_values('dED');cv=pd.read_csv(SUPP/'landmark_subset_size_nestedCV.csv')
 C.panel(4,6,'a','Landmark ablation');C.panel(94,6,'b','Nested-CV subset size')
 a=C.ax(20,16,64,40);a.barh(ab.landmark,ab.dED,color=[PURPLE if v>0 else LIGHT for v in ab.dED],height=.65)
 a.axvline(0,c=INK,lw=.6);a.set(xlabel='Δ CV median ED when removed (mm)');tidy(a)
 a=C.ax(106,16,68,40);a.fill_between(cv.k,cv.q25,cv.q75,color=PURPLE,alpha=.15,lw=0);a.plot(cv.k,cv.median_outer_ED,'-o',c=PURPLE,ms=3,lw=1)
 a.set(xlabel='Landmarks selected (greedy forward, inner CV)',ylabel='Outer-fold ED (mm)',xticks=cv.k);tidy(a)
 C.t(5,66,'a: removal of each landmark from the 11-landmark model (4-fold grouped CV). b: subset chosen inside each outer training fold, scored on the held-out fold (line: median; band: IQR).',6.0,c=MUTE)
 C.save('Figure_S6')

# 16-AFID protocol reference: real MNI152NLin2009bAsym landmark positions
# (afids-pred/figdata/mni_landmarks.csv, extracted from the ground-truth
# annotation on that template) shown on the real T1 template in the three
# canonical planes, plus an index of every abbreviation/number/full name.
AFID_META=[(1,'AC','M','Anterior commissure'),(2,'PC','M','Posterior commissure'),(3,'ICS','M','Infracollicular sulcus'),
 (4,'PMJ','M','Pontomesencephalic junction'),(5,'SIPF','M','Superior interpeduncular fossa'),
 (6,'SLMS','R','Right superior lateral mesencephalic sulcus'),(7,'SLMS','L','Left superior lateral mesencephalic sulcus'),
 (8,'ILMS','R','Right inferior lateral mesencephalic sulcus'),(9,'ILMS','L','Left inferior lateral mesencephalic sulcus'),
 (11,'IMS','M','Intermammillary sulcus'),(12,'MB','R','Right mammillary body'),(13,'MB','L','Left mammillary body'),
 (15,'LVAC','R','Right lateral ventricle at AC'),(16,'LVAC','L','Left lateral ventricle at AC'),
 (17,'LVPC','R','Right lateral ventricle at PC'),(18,'LVPC','L','Left lateral ventricle at PC')]
REGION_COLORS={'commissural':INK,'midbrain':PURPLE,'brainstem':TEAL,'diencephalic':GOLD}

def supp1():
 C=Canvas(150);L=pd.read_csv(NEW/'mni_landmarks.csv')
 im,vol=anatomy()
 rows=[]
 for idx,abbr,hemi,full in AFID_META:
  g=L[(L.landmark==abbr)&(L.hemi==hemi)];assert len(g)==1,(idx,abbr,hemi);r=g.iloc[0]
  rows.append(dict(idx=idx,abbr=abbr,hemi=hemi,full=full,region=r.region,x=r.x,y=r.y,z=r.z))
 pts=pd.DataFrame(rows)
 C.panel(4,7,'a','16-AFID protocol subset, real MNI152NLin2009bAsym anatomy')
 views=[('Sagittal',0,'y','z',(-45,35),(-40,45)),('Coronal',1,'x','z',(-25,25),(-40,45)),('Axial',2,'x','y',(-25,25),(-45,35))]
 for title,slot,h1,h2,lim1,lim2 in views:
  a=C.ax(6+slot*58,18,52,52)
  gg1,gg2=np.meshgrid(np.linspace(*lim1,260),np.linspace(*lim2,260))
  if title=='Sagittal':xx,yy,zz=np.zeros_like(gg1),gg1,gg2
  elif title=='Coronal':xx,yy,zz=gg1,np.full_like(gg1,pts.y.mean()),gg2
  else:xx,yy,zz=gg1,gg2,np.full_like(gg1,pts.z.mean())
  vv=sample(im,vol,xx,yy,zz);a.imshow(vv,origin='lower',extent=(*lim1,*lim2),cmap='gray',vmin=0,vmax=np.percentile(vv,99.5));a.axis('off')
  a.set_title(title,fontsize=7,pad=3)
  placed=[]
  for _,r in pts.iterrows():
   c=REGION_COLORS[r.region];p1,p2=(r[h1],r[h2])
   a.scatter(p1,p2,s=20,c=c,edgecolor='white',lw=.5,zorder=5)
   near=sum(np.hypot(p1-q1,p2-q2)<4 for q1,q2 in placed);placed.append((p1,p2))
   off=[(2.2,2.2),(-6.5,2.2),(2.2,-6.0),(-6.5,-6.0)][near%4]
   a.annotate(str(r.idx),(p1,p2),xytext=off,textcoords='offset points',fontsize=4.6,color=c,weight='bold',zorder=6)
 for i,(region,c) in enumerate(REGION_COLORS.items()):
  C.a.scatter([6+i*30],[73],s=18,c=c);C.t(9+i*30,73,region,6,c=INK)
 C.panel(4,80,'b','Landmark index')
 for i,r in enumerate(rows):
  col_=i//8;row_=i%8;x0=6+col_*88;yy=94+row_*7
  C.t(x0,yy,f'{r["idx"]}',6.3,c=REGION_COLORS[r['region']],bold=True)
  C.t(x0+7,yy,r['abbr']+('' if r['hemi']=='M' else f'-{r["hemi"]}'),6.3,bold=True)
  C.t(x0+22,yy,r['full'],6.3,c=INK)
 C.t(4,148,'Positions: AFIDs ground-truth annotation on the MNI152NLin2009bAsym template (Lau et al., 2019).',6.2,c=MUTE)
 C.save('Figure_S1')

if __name__=='__main__':
 for name in sys.argv[1:] or ['1','2','3','4','S1','S2','S3','S4','S5','S6']:{'1':fig1,'2':fig2,'3':fig3,'4':fig4,'S1':supp1,'S2':supp3,'S3':supp2,'S4':supp5,'S5':supp4,'S6':supp6}[name]()
