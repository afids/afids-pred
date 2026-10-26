from pathlib import Path
import sys,json,warnings
import os
# ROOT = this repository. Template volumes / atlas meshes are not redistributed here; they are read
# from AFIDS_FIG_EXTERNAL (default: the directory that contains this repository).
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT;EXT=Path(os.environ.get('AFIDS_FIG_EXTERNAL',ROOT.parent))
OLD=EXT/'manuscript_figure_new';OUT=ROOT/'figures/manuscript';D=ROOT/'figures/source_data';PREVIEW=ROOT/'figures/.preview'
sys.path.insert(0,str(OLD/'.render_deps'));warnings.filterwarnings('ignore')
import numpy as np,pandas as pd,nibabel as nib
from scipy.ndimage import map_coordinates
from scipy.stats import wilcoxon
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.patches import Rectangle,FancyArrowPatch
from PIL import Image
BLUE='#226A96';GOLD='#C57932';INK='#20282D';MUTE='#65717A';PALE='#D9E6ED';GRAY='#A1ABB3';RULE='#D5DBDF'
DIV=LinearSegmentedColormap.from_list('signed',['#25618A','#A5C3D6','#FCFCF8','#E1B694','#B86032'])
SEQ=LinearSegmentedColormap.from_list('weight',['#D3E4ED','#5595B7','#0E4E77'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.labelsize':7,'xtick.labelsize':6.3,'ytick.labelsize':6.3,'axes.linewidth':.55,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'savefig.facecolor':'white','pdf.fonttype':42,'svg.fonttype':'none'})
class Canvas:
 def __init__(self,h):
  self.h=h;self.f=plt.figure(figsize=(180/25.4,h/25.4));self.a=self.f.add_axes([0,0,1,1]);self.a.set(xlim=(0,180),ylim=(h,0));self.a.axis('off');self.a.patch.set_visible(False);self.a.set_zorder(30)
 def ax(self,x,y,w,h,**kw):
  ax=self.f.add_axes([x/180,1-(y+h)/self.h,w/180,h/self.h],**kw);ax.patch.set_alpha(0);return ax
 def t(self,x,y,s,size=7,c=INK,bold=False,ha='left',**kw):self.a.text(x,y,s,fontsize=size,color=c,weight='bold' if bold else 'normal',ha=ha,va='center',**kw)
 def panel(self,x,y,l,s):self.t(x,y,l,10,bold=True);self.t(x+6,y,s,8.4,bold=True)
 def line(self,x,y,xx,yy,c=RULE,lw=.6):self.a.plot([x,xx],[y,yy],color=c,lw=lw)
 def arrow(self,x,y,xx,yy,c=MUTE):self.a.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=7,color=c,lw=.75))
 def save(self,name):
  self.f.savefig(OUT/f'{name}.png',dpi=610)
  PREVIEW.mkdir(exist_ok=True);im=Image.open(OUT/f'{name}.png');im.resize((1260,round(1260*self.h/180))).save(PREVIEW/f'{name}.png');print(name,im.size,flush=True)

def tidy(ax):
 ax.tick_params(width=.55,length=2.5);ax.spines['left'].set_color(MUTE);ax.spines['bottom'].set_color(MUTE)
 ax.set_axisbelow(True)

def anatomy():
 p=OLD/'reference_review/mni/mni_icbm152_nlin_asym_09b/mni_icbm152_t1_tal_nlin_asym_09b_hires.nii';im=nib.load(p);return im,np.asanyarray(im.dataobj).astype('float32')
def sample(im,vol,x,y,z):
 shape=np.asarray(x).shape;v=nib.affines.apply_affine(np.linalg.inv(im.affine),np.c_[np.ravel(x),np.ravel(y),np.ravel(z)]).T
 return map_coordinates(vol,v,order=1,mode='constant').reshape(shape)

from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource
TEAL="#008C95";CORAL="#ED624C";PURPLE="#7052B6";LIGHT="#D8DDE1"

def paired(ax,df,left,right,metric,colors,labels,ylim=None):
 pp=df.pivot(index=['subject','hemisphere'],columns='group',values=metric)[[left,right]].dropna()
 for idx,row in pp.iterrows():
  ax.plot([0,1],row.values,color='#C4CDD3',lw=.55,zorder=1)
  ax.scatter([0,1],row.values,s=10,c=colors,edgecolors='white',linewidths=.25,zorder=3)
 for j,key in enumerate([left,right]):
  lo,med,hi=np.percentile(pp[key],[25,50,75]);ax.plot([j,j],[lo,hi],color=colors[j],lw=3.0,zorder=4);ax.plot([j-.12,j+.12],[med,med],c=INK,lw=1.15,zorder=5)
 ax.set_xlim(-.3,1.3);ax.set_xticks([0,1]);ax.set_xticklabels(labels);tidy(ax)
 if ylim:ax.set_ylim(ylim)
 return pp

def plane(ax,im,vol,kind,bounds,fixed,alpha=1,res=160):
 aa,bb=np.meshgrid(np.linspace(bounds[0],bounds[1],res),np.linspace(bounds[2],bounds[3],res))
 if kind=='sag':x=np.full_like(aa,fixed);y=aa;z=bb
 elif kind=='cor':x=aa;y=np.full_like(aa,fixed);z=bb
 else:x=aa;y=bb;z=np.full_like(aa,fixed)
 v=sample(im,vol,x,y,z);rgba=plt.cm.gray(np.clip(v/np.percentile(v,99.5),0,1)**1.2);rgba[...,3]=alpha
 ax.plot_surface(x,y,z,facecolors=rgba,rstride=1,cstride=1,shade=False,antialiased=False,zorder=1)

def surfaces(ax,names=('STN','RN')):
 for hemi in ['lh','rh']:
  for name in names:
   im=nib.load(OLD/'atlas_sources'/f'{hemi}_{name}.nii.gz');v=im.get_fdata();verts,faces,_,_=marching_cubes(gaussian_filter(v,.7),.5,step_size=2);verts=nib.affines.apply_affine(im.affine,verts)
   ax.add_collection3d(Poly3DCollection(verts[faces],facecolors=CORAL if name=='STN' else '#B8C2CC',edgecolor='none',shade=True,lightsource=LightSource(azdeg=300,altdeg=50),zorder=4 if name=='STN' else 3))

def cloud(ax,values,x,c):
 vals=np.asarray(values);rng=np.random.default_rng(24)
 vp=ax.violinplot([vals],positions=[x],widths=.58,showextrema=False,bw_method=.4)
 for b in vp['bodies']:b.set_facecolor(c);b.set_edgecolor('none');b.set_alpha(.15)
 ax.scatter(x+rng.uniform(-.12,.12,len(vals)),vals,s=10,c=c,alpha=.8,edgecolor='white',linewidth=.3,zorder=3)
 lo,med,hi=np.percentile(vals,[25,50,75]);ax.plot([x,x],[lo,hi],lw=3,c=c,zorder=4);ax.scatter(x,med,s=19,c='white',edgecolor=c,zorder=5)

base=sys.modules[__name__]

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
