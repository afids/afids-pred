"""Same treatment as export_panels.py (Figure 1), applied to Figures 2-4:
every axes in each figure, rendered as its own standalone PNG at the same
physical mm size it has in the composited figure, using the exact plotting
calls currently in render_figures.py -- no baked-in narration/captions, so
these are raw pieces for manual reassembly.

Run: python figures/source/export_panels_2345.py   (needs the external template volumes, see docs/REPRODUCIBILITY.md)
Out: figures/panels/fig2/, fig3/, fig4/
"""
from render_figures import *  # noqa: F401,F403
from export_panels import PANELS  # reuse the same panels/ root

for sub in ['fig2', 'fig3', 'fig4']:
    (PANELS / sub).mkdir(exist_ok=True)


def save(fig, sub, name, transparent=True):
    fig.savefig(PANELS / sub / f'{name}.png', dpi=610, bbox_inches='tight', pad_inches=0.03, transparent=transparent)
    fig.savefig(PANELS / sub / f'{name}.pdf', dpi=610, bbox_inches='tight', pad_inches=0.03, transparent=transparent)
    plt.close(fig)
    print('saved', sub, name)


def newfig(w_mm, h_mm, **kw):
    fig = plt.figure(figsize=(w_mm / 25.4, h_mm / 25.4))
    ax = fig.add_axes([0, 0, 1, 1], **kw)
    return fig, ax


# ================================================================= Figure 2
def fig2_panels():
    corr = pd.read_csv(BASE / 'fig2_correlations.csv', index_col=0)
    tc = pd.read_csv(BASE / 'fig2_stn_correlations.csv', index_col=0)
    order = [f for name in names for f in corr.index if f[:-1] == name]
    lab = [f[:-1] + ' ' + f[-1] for f in order]
    bounds = [i for i in range(1, len(order)) if order[i][:-1] != order[i - 1][:-1]]

    fig, a = newfig(71, 71)
    h = a.imshow(corr.loc[order, order], cmap=DIV, vmin=-1, vmax=1)
    a.set_xticks(range(28)); a.set_xticklabels(lab, rotation=90, fontsize=5.7)
    a.set_yticks(range(28)); a.set_yticklabels(lab, fontsize=5.7); a.tick_params(length=0, pad=1)
    for sp in a.spines.values(): sp.set_visible(False)
    for k in bounds: a.axhline(k - .5, c='white', lw=.8); a.axvline(k - .5, c='white', lw=.8)
    save(fig, 'fig2', 'a_correlation_matrix')

    fig, a = newfig(9, 71)
    a.imshow(tc.loc[order], cmap=DIV, vmin=-1, vmax=1, aspect='auto')
    a.set(xticks=[0, 1, 2], xticklabels=['x', 'y', 'z'], yticks=[]); a.set_title('STN', fontsize=6, pad=3); a.tick_params(length=0, pad=1)
    for sp in a.spines.values(): sp.set_visible(False)
    save(fig, 'fig2', 'a_stn_correlation_strip')

    fig, cax = newfig(50, 2)
    sm_ = plt.cm.ScalarMappable(cmap=DIV, norm=Normalize(-1, 1))
    cb = fig.colorbar(sm_, cax=cax, orientation='horizontal')
    cb.set_ticks([-1, 0, 1]); cb.outline.set_visible(False); cb.ax.tick_params(length=2, labelsize=6, pad=1)
    save(fig, 'fig2', 'a_colorbar')

    # 3D PCA scatter: light box + grid (not the default heavy grey) + a
    # floor-projected shadow for depth -- matches fig2()'s current style.
    # Unlike newfig()'s full-bleed [0,0,1,1] axes, this one gets a 20%
    # margin for the rotated axis labels. bbox_inches='tight' (used by the
    # shared save() below) actively mis-crops this specific plot -- its
    # extent estimate for Axes3D's rotated/projected tick labels doesn't
    # match final render, clipping "PC1"/"PC3" regardless of margin size
    # (confirmed: identical figure renders correctly without tight bbox) --
    # so this one export skips save() and uses a fixed bbox instead.
    fig = plt.figure(figsize=(80 / 25.4, 78 / 25.4))
    a = fig.add_axes([0.2, 0.2, 0.6, 0.6], projection='3d', computed_zorder=False)
    zlo = -8
    a.scatter(scores[:, 0], scores[:, 1], np.full(len(scores), zlo), s=2, c=LIGHT, alpha=.25, depthshade=False, zorder=1)
    a.scatter(scores[:, 0], scores[:, 1], scores[:, 2], s=10, c=scores[:, 0], cmap='Purples', alpha=.8, edgecolor='white', linewidth=.15, depthshade=False, zorder=5)
    a.set(xlim=(-8, 6), ylim=(-8, 6), zlim=(zlo, 6))
    a.view_init(23, -57); a.set_proj_type('ortho'); a.set_box_aspect([1, 1, 1], zoom=1.25)
    a.set_xlabel(f'PC1 · {pca.explained_variance_ratio_[0]:.0%}', labelpad=-3, fontsize=6.3, color=MUTE)
    a.set_ylabel(f'PC2 · {pca.explained_variance_ratio_[1]:.0%}', labelpad=-3, fontsize=6.3, color=MUTE)
    a.set_zlabel(f'PC3 · {pca.explained_variance_ratio_[2]:.0%}', labelpad=-5, fontsize=6.3, color=MUTE)
    a.tick_params(labelsize=5, pad=-2, colors=MUTE); a.grid(True, color=LIGHT, linewidth=.4)
    for axis in [a.xaxis, a.yaxis, a.zaxis]: axis.pane.fill = False; axis.pane.set_edgecolor(LIGHT); axis.line.set_color(LIGHT)
    fig.savefig(PANELS / 'fig2' / 'b_pca_3d_scatter.png', dpi=610, transparent=True)
    plt.close(fig); print('saved fig2 b_pca_3d_scatter')

    fig, a = newfig(50, 22)
    cum = pca.explained_variance_ratio_.cumsum()
    a.fill_between(range(1, 34), cum * 100, color=PURPLE, alpha=.12, lw=0)
    a.plot(range(1, 34), cum * 100, c=PURPLE, lw=1.3); a.scatter(23, cum[22] * 100, c=CORAL, s=23, zorder=5)
    a.axhline(99, c=LIGHT, lw=.55, ls='--'); a.text(11, 38, '23 PCs / 99%', fontsize=7, color=PURPLE)
    a.set(xlim=(1, 33), ylim=(0, 105), xticks=[1, 16, 23, 33], yticks=[0, 50, 99], xlabel='Components', ylabel='Variance (%)'); tidy(a)
    save(fig, 'fig2', 'b_variance_explained')

    # PC1/PC2 mode plots: real midline-sagittal FLASH anatomy behind the
    # mode-shift lines, matching fig2()'s current style exactly.
    flash_im = nib.load(FLASH_PATH); flash_vol = np.asanyarray(flash_im.dataobj, dtype='float32')
    gy, gz = np.meshgrid(np.linspace(-34, 20, 360), np.linspace(-25, 38, 360))
    npts = native(np.column_stack([np.zeros(gy.size), gy.ravel(), gz.ravel()]))
    sag = sample(flash_im, flash_vol, npts[:, 0].reshape(gy.shape), npts[:, 1].reshape(gy.shape), npts[:, 2].reshape(gy.shape))
    for mode in range(2):
        fig, a = newfig(38, 34)
        a.imshow(sag, origin='lower', extent=(-34, 20, -25, 38), cmap='gray', vmin=np.percentile(sag, 1), vmax=np.percentile(sag, 99), zorder=0)
        dire = np.zeros((2, 33)); dire[:, mode] = np.array([-2, 2]) * np.sqrt(pca.explained_variance_[mode])
        vals = sc.inverse_transform(pca.inverse_transform(dire))
        for name in names:
            ids = [X.columns.get_loc(name + k) for k in 'xyz']; v = vals[:, ids]
            a.plot(v[:, 1], v[:, 2], c=col[name], lw=1.4, zorder=3, path_effects=[pe.withStroke(linewidth=2.4, foreground='white')])
            a.scatter(v[0, 1], v[0, 2], s=16, facecolors='white', edgecolors=[col[name]], linewidth=.8, zorder=4)
            a.scatter(v[1, 1], v[1, 2], s=16, c=[col[name]], linewidth=0, zorder=4)
        a.set(xlim=(-34, 20), ylim=(-25, 38), xticks=[-20, 0, 20], yticks=[-20, 0, 20], xlabel='y (mm)', ylabel='z (mm)' if mode == 0 else ''); tidy(a)
        save(fig, 'fig2', f'c_mode_PC{mode + 1}')

    # Landmark color key (11 swatches) -- panel c's own legend, drawn
    # separately in fig2() rather than baked into either mode plot.
    fig, ax = newfig(88, 4)
    ax.set_xlim(0, 88); ax.set_ylim(4, 0); ax.axis('off')
    for i, name in enumerate(names):
        x0 = i * 7.9
        ax.add_patch(Rectangle((x0, 1), 2, 2, fc=col[name], ec='none')); ax.text(x0 + 2.7, 2, name, fontsize=9, color=INK, va='center')
    save(fig, 'fig2', 'c_landmark_legend')

    weights = (abs(pca.components_) * pca.explained_variance_ratio_[:, None]).sum(0)
    wt = pd.Series(weights, index=X.columns)
    arr = np.array([[wt[name + k] for k in 'xyz'] for name in names])
    fig, a = newfig(24, 42)
    im = a.imshow(arr, cmap=SEQ, aspect='auto', vmin=0, vmax=arr.max())
    a.set(xticks=[0, 1, 2], xticklabels=['x', 'y', 'z'], yticks=range(11), yticklabels=names); a.tick_params(length=0, labelsize=6)
    for sp in a.spines.values(): sp.set_visible(False)
    save(fig, 'fig2', 'd_contributions_heatmap')

    fig, cax = newfig(2, 42)
    sm_ = plt.cm.ScalarMappable(cmap=SEQ, norm=Normalize(0, arr.max()))
    cb = fig.colorbar(sm_, cax=cax); cb.outline.set_visible(False); cb.ax.tick_params(length=2, labelsize=5.8)
    save(fig, 'fig2', 'd_colorbar')


# ================================================================= Figure 3
def fig3_panels():
    for ds, c, lab in zip(DS, DC, DL):
        fig, a = newfig(32, 34)
        residual_axis(a, data[(data.dataset == ds) & (data.method == 'AFIDs')], c, lab)
        a.set_xlabel('Δy (mm)', fontsize=6.3); a.set_ylabel('Δz (mm)', fontsize=6.3)
        save(fig, 'fig3', f'a_residuals_{ds}')

    fig, a = newfig(73, 36)
    for ds, c, lab in zip(DS, DC, ['Clinical 1.5 T + gad', 'Clinical 7 T', 'External 3 T', 'External 7 T']):
        ecdf(a, data[(data.dataset == ds) & (data.method == 'AFIDs')].ED, c, lab)
    a.set(xlim=(0, 2.3), ylim=(0, 1.03), xticks=[0, 1, 2], yticks=[0, .5, 1], xlabel='STN error (mm)', ylabel='Fraction'); tidy(a)
    a.legend(loc='lower right', fontsize=5.5, frameon=False, handlelength=1.2, labelspacing=.2)
    save(fig, 'fig3', 'b_ecdf')

    fig, a = newfig(50, 36)
    for j, (d1, d2) in enumerate([(DS[0], DS[1]), (DS[2], DS[3])]):
        pp = data[(data.method == 'AFIDs') & data.dataset.isin([d1, d2])].pivot(index=['subject', 'hemisphere'], columns='dataset', values='ED')
        v = pp[d1] - pp[d2]; cloud(a, v, j, DC[j * 2]); pv = wilcoxon(np.round(v, 3)).pvalue
        a.text(j, 1.25, f'p = {pv:.2f}', ha='center', fontsize=6)
    a.axhline(0, c=INK, lw=.65, ls='--')
    a.set(xlim=(-.6, 1.6), ylim=(-1.4, 1.5), yticks=[-1, 0, 1], xticks=[0, 1], xticklabels=['1.5 T − 7 T', '3 T − 7 T'], ylabel='Δ error (mm)'); tidy(a)
    save(fig, 'fig3', 'c_paired_contrast')

    fig, a = newfig(73, 21)
    for j, ds in enumerate(DS[:2]):
        dd = data[data.dataset == ds].pivot(index=['subject', 'hemisphere'], columns='method', values='ED')
        c = DC[j]; xs = [j * 2.4, j * 2.4 + 1]
        for _, r in dd.iterrows():
            a.plot(xs, [r.Registration, r.AFIDs], c=LIGHT, lw=.5)
            a.scatter(xs, [r.Registration, r.AFIDs], s=8, c=['#8D9AA4', c], edgecolor='white', lw=.2, zorder=3)
        for xx, v, cc in [(xs[0], dd.Registration, '#8D9AA4'), (xs[1], dd.AFIDs, c)]:
            lo, med, hi = np.percentile(v, [25, 50, 75]); a.plot([xx, xx], [lo, hi], c=cc, lw=3); a.scatter(xx, med, s=15, c='white', edgecolor=cc, zorder=5)
    a.set(xticks=[0, 1, 2.4, 3.4], xticklabels=['Atlas', 'AFIDs', 'Atlas', 'AFIDs'], ylim=(0, 3.5), yticks=[0, 1.5, 3], ylabel='Error (mm)'); tidy(a)
    save(fig, 'fig3', 'd_registration_benchmark')

    fig, a = newfig(50, 21)
    for j, ds in enumerate(DS[:2]):
        dd = data[data.dataset == ds].pivot(index=['subject', 'hemisphere'], columns='method', values='ED')
        v = dd.Registration - dd.AFIDs; cloud(a, v, j, DC[j]); pv = wilcoxon(v).pvalue
        a.text(j, 2.33, 'p = ' + ('0.00085' if j == 0 else '0.60'), fontsize=5.8, ha='center')
    a.axhline(0, c=INK, ls='--', lw=.6)
    a.set(xticks=[0, 1], xticklabels=['1.5 T + gad', '7 T'], ylim=(-1, 2.8), yticks=[-1, 0, 1, 2], ylabel='Atlas − AFIDs (mm)'); tidy(a)
    save(fig, 'fig3', 'e_paired_gain')


# ================================================================= Figure 4
def fig4_panels():
    ann = pd.read_csv(NEW / 'annotation_predictions.csv')
    secondary = pd.read_csv(NEW / 'secondary_targets.csv')

    # Real 3D STN surface model (bilateral, DISTAL atlas) with actual
    # prediction points scattered around it, on a real 200-um FLASH25 axial
    # plane (same template as Figure 1/2) in native template space.
    fig, a = newfig(100, 72, projection='3d', computed_zorder=False)
    hemi_pts = {}
    for h in ['R', 'L']:
        dd = ann[ann.hemisphere == h]; v = dd[['pred_' + k for k in 'xyz']].to_numpy(); gt = dd[['truth_' + k for k in 'xyz']].iloc[0].to_numpy()
        if h == 'L': v[:, 0] *= -1; gt[0] *= -1
        hemi_pts[h] = (native(v), native(gt))
    zfix = float(np.mean([hemi_pts['R'][1][2], hemi_pts['L'][1][2]]))
    flash_im = nib.load(FLASH_PATH); flash_vol = np.asanyarray(flash_im.dataobj, dtype='float32')
    plane(a, flash_im, flash_vol, 'ax', [-23, 23, -25, 3], zfix, res=200); atlasmesh(a, .28)
    for h, (v, gt) in hemi_pts.items():
        for pt in v: a.plot([gt[0], pt[0]], [gt[1], pt[1]], [gt[2], pt[2]], c=CORAL, alpha=.22, lw=.5, zorder=3)
        a.scatter(*v.T, s=13, c=CORAL, alpha=.8, edgecolor='white', lw=.3, depthshade=False, zorder=6)
        a.scatter(*gt, s=48, marker='*', c=INK, edgecolor='white', lw=.6, depthshade=False, zorder=7)
    frame3(a, [(-23, 23), (-25, 3), (-22, 4)], view=(90, -90), zoom=1.55)
    save(fig, 'fig4', 'a_stn_model_3d')

    fig, a = newfig(51, 55)
    xx = np.linspace(ann.AFID_error.min(), ann.AFID_error.max(), 100)
    fit = sm.OLS(ann.ED, sm.add_constant(ann.AFID_error)).fit()
    pr = fit.get_prediction(sm.add_constant(xx)).summary_frame()
    a.fill_between(xx, pr.mean_ci_lower, pr.mean_ci_upper, color='#D7ECEC', zorder=0)
    for h, m in [('R', 'o'), ('L', 's')]:
        dd = ann[ann.hemisphere == h]; a.scatter(dd.AFID_error, dd.ED, s=15, c=TEAL, marker=m, alpha=.75, edgecolor='white', lw=.3, zorder=3)
    a.plot(xx, pr['mean'], c=INK, lw=1)
    a.set(xlabel='AFID error (mm)', ylabel='STN error (mm)', xticks=[.5, 1, 1.5, 2], yticks=[.5, 1, 1.5, 2, 2.5]); tidy(a)
    a.text(.05, .96, 'r = 0.47\nSlope = 0.46', transform=a.transAxes, va='top', fontsize=7)
    save(fig, 'fig4', 'b_afid_vs_stn_error')

    fig, a = newfig(70, 42)
    tcols = {'cZI': PURPLE, 'fct': '#CA9528', 'RN': TEAL}
    for target, c in tcols.items():
        v = secondary[secondary.target == target].ed; ecdf(a, v, c, f'{target}  {np.median(v):.2f} mm')
    a.set(xlim=(0, max(secondary.ed) * 1.02), ylim=(0, 1.03), xticks=[0, 1, 2, 3], yticks=[0, .5, 1], xlabel='Target error (mm)', ylabel='Fraction')
    tidy(a); a.legend(loc='lower right', fontsize=6, frameon=False, handlelength=1.3, title='Median', title_fontsize=6)
    save(fig, 'fig4', 'c_ecdf_secondary_targets')

    models = ['Ridge', 'BayesianRidge', 'ElasticNet', 'Lasso', 'PLS', 'RandomForest', 'XGBoost', 'SVR', 'KNN', 'MLP']
    arr = np.empty((10, 3))
    for j, target in enumerate(['cZI', 'fct', 'RN']):
        arr[:, j] = pd.read_csv(REPO / f'data/{target}_logocv_results_unified.csv').groupby('model').ed.median().reindex(models)
    fig, a = newfig(23, 49)
    h = a.imshow(arr, cmap='magma_r', aspect='auto', vmin=0, vmax=3)
    a.set(xticks=[0, 1, 2], xticklabels=['cZI', 'fct', 'RN'], yticks=range(10), yticklabels=models); a.tick_params(length=0, labelsize=6)
    for sp in a.spines.values(): sp.set_visible(False)
    for j in range(3):
        i = np.argmin(arr[:, j]); a.add_patch(Rectangle((j - .47, i - .47), .94, .94, fill=False, ec=TEAL, lw=1.1))
    save(fig, 'fig4', 'd_architecture_screen_heatmap')

    fig, cax = newfig(2, 49)
    sm_ = plt.cm.ScalarMappable(cmap='magma_r', norm=Normalize(0, 3))
    cb = fig.colorbar(sm_, cax=cax); cb.set_label('Median error (mm)', fontsize=6, labelpad=3); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.8, length=2)
    save(fig, 'fig4', 'd_colorbar')


if __name__ == '__main__':
    fig2_panels()
    fig3_panels()
    fig4_panels()
    print('all panels exported under', PANELS)
