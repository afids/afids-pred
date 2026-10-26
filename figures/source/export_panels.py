"""Export every visual component of Figure 1 as its own standalone PNG, so
they can be reassembled/repositioned by hand in PowerPoint/Illustrator/etc.
instead of via this script's own (repeatedly contested) layout math.

Reuses the exact data and drawing calls from render_figures.fig1() -- same
numbers, same colors, same real anatomy -- just each piece rendered to its
own tightly-cropped figure instead of one composited canvas.

Run: python figures/source/export_panels.py   (needs the external template volumes, see docs/REPRODUCIBILITY.md)
Out: figures/panels/*.png
"""
from pathlib import Path
from render_figures import *  # noqa: F401,F403 -- reuse all loaded data/helpers

PANELS = REPO / 'figures/panels'
PANELS.mkdir(exist_ok=True)


def save(fig, name, transparent=True):
    fig.savefig(PANELS / f'{name}.png', dpi=610, bbox_inches='tight', pad_inches=0.03, transparent=transparent)
    fig.savefig(PANELS / f'{name}.pdf', dpi=610, bbox_inches='tight', pad_inches=0.03, transparent=transparent)
    plt.close(fig)
    print('saved', name)


def newfig(w_mm, h_mm, **kw):
    # fig.add_axes([0,0,1,1]) makes the axes span the WHOLE figure, so 1 data
    # unit = 1 mm physical -- matching drawing.Canvas exactly. plt.subplots()
    # instead leaves matplotlib's default subplot margins in place, silently
    # shrinking the usable data area; every font size / position tuned to
    # match fig1() would then overlap or clip against those margins.
    fig = plt.figure(figsize=(w_mm / 25.4, h_mm / 25.4))
    ax = fig.add_axes([0, 0, 1, 1], **kw)
    return fig, ax


def load_targets():
    sec = pd.read_csv(NEW / 'secondary_target_coords.csv')
    stn = pd.read_csv(NEW / 'development_stn.csv', index_col=0)
    target_allv = {}
    for target in TARGET_COLORS:
        allv = []
        for h in ['R', 'L']:
            if target == 'STN':
                v = stn[[h + 'STN' + k for k in 'xyz']].to_numpy()
                if h == 'L': v[:, 0] *= -1
            else:
                v = sec[(sec.target == target) & (sec.hemisphere == h)][['x', 'y', 'z']].to_numpy()
            allv.append(v)
        target_allv[target] = np.concatenate(allv)
    return target_allv


# ============================================================= panel a: 3D scatter
def export_3d_scatter():
    # No circle baked in here (per earlier explicit request) -- this piece is
    # for manual placement, so the circle+callout geometry is left to hand
    # assembly. Size/zoom match fig1()'s current 50x78mm panel exactly.
    target_allv = load_targets()
    fig = plt.figure(figsize=(50 / 25.4, 78 / 25.4))
    a = fig.add_axes([0, 0, 1, 1], projection='3d', computed_zorder=False)
    means = []
    for name in names:
        v = X[[name + k for k in 'xyz']].to_numpy(); mu = v.mean(0); means.append(mu)
        a.scatter(v[:, 0], v[:, 1], np.full(len(v), -28), s=1, c=MUTE, alpha=.05, depthshade=False, zorder=1)
        a.scatter(*v.T, s=3.0, c=MUTE, alpha=.28, depthshade=False, zorder=5)
        w, V = np.linalg.eigh(np.cov(v.T)); w = np.maximum(w, 0); tt = np.linspace(0, 2 * np.pi, 60)
        for j, k in [(0, 1), (1, 2), (0, 2)]:
            ring = mu + np.cos(tt)[:, None] * np.sqrt(w[j]) * V[:, j] + np.sin(tt)[:, None] * np.sqrt(w[k]) * V[:, k]
            poly = Poly3DCollection([ring], facecolor=MUTE, edgecolor=MUTE, lw=.4, alpha=.14, zorder=2)
            a.add_collection3d(poly)
        a.plot([mu[0], mu[0]], [mu[1], mu[1]], [mu[2], -28], c=LIGHT, lw=.6, ls=(0, (1.5, 1.5)), zorder=1)
        a.scatter(*mu, s=28, c=INK, edgecolor='white', lw=.7, depthshade=False, zorder=8)
    for xx in [0, 10, 20, 30]: a.plot([xx, xx], [-37, 23], [-28, -28], c=LIGHT, lw=.4, zorder=0)
    for yy in [-30, -20, -10, 0, 10, 20]: a.plot([0, 33], [yy, yy], [-28, -28], c=LIGHT, lw=.4, zorder=0)
    frame3(a, [(-3, 34), (-38, 24), (-29, 40)], view=(21, 38), zoom=1.35)
    for target, c in TARGET_COLORS.items():
        for v, m in zip(np.split(target_allv[target], 2), ['o', 's']):
            a.scatter(*v.T, s=7, c=c, marker=m, alpha=.8, edgecolor='white', lw=.18, depthshade=False, zorder=9)

    save(fig, 'panel_a_3d_scatter')


# ============================================================= lens/magnifier
def export_lens():
    # Plain black rim -- fig1() dropped the bezel/shadow/vignette/scale bar.
    flash_im = nib.load(FLASH_PATH); flash_vol = np.asanyarray(flash_im.dataobj, dtype='float32')
    z = float(geom['targets_native'][:, 2].mean()); cx, cy = 9, -17; mag_r = 20
    target_allv = load_targets()
    xx, yy = np.meshgrid(np.linspace(cx - mag_r, cx + mag_r, 640), np.linspace(cy - mag_r, cy + mag_r, 640))
    zz = np.full_like(xx, z); vv = sample(flash_im, flash_vol, xx, yy, zz)

    fig, b = newfig(50, 50)
    b.set_xlim(cx - mag_r, cx + mag_r); b.set_ylim(cy - mag_r, cy + mag_r); b.axis('off')
    clip = Circle((cx, cy), mag_r, transform=b.transData)
    im = b.imshow(vv, origin='lower', extent=(cx - mag_r, cx + mag_r, cy - mag_r, cy + mag_r), cmap='gray',
                  vmin=np.percentile(vv, 1), vmax=np.percentile(vv, 99.4)); im.set_clip_path(clip)
    for target, c in [('STN', CORAL), ('RN', '#CBD3D8'), ('cZI', PURPLE), ('fct', '#CA9528')]:
        ai = seg_mask('R', target); av = sample(ai, ai.get_fdata(), xx, yy, zz)
        cont = b.contour(xx, yy, av, levels=[.5], colors=[c], linewidths=1.3); cont.set_clip_path(clip)
    for target, c in TARGET_COLORS.items():
        for v, m in zip(np.split(target_allv[target], 2), ['o', 's']):
            nv = native(v); scat = b.scatter(nv[:, 0], nv[:, 1], s=9, c=c, marker=m, alpha=.75, edgecolor='white', lw=.25, zorder=5)
            scat.set_clip_path(clip)
    b.add_patch(Circle((cx, cy), mag_r, fill=False, ec='black', lw=2.2, zorder=7))
    save(fig, 'panel_b_lens', transparent=False)  # opaque: anatomy fills the whole circle/box


# ======================================== ground-truth-annotation venn diagram
def export_venn():
    fig, ax = newfig(24, 24)
    venn_majority(ax)
    save(fig, 'ground_truth_venn')


# ============================== rater/overlap legend (drawn beside the venn in fig1())
def export_rater_legend():
    raters = [('Rater 1', RATER_COLORS[1]), ('Rater 2', RATER_COLORS[2]), ('Rater 3', RATER_COLORS[0]), ('Overlap', CORAL)]
    fig, ax = newfig(20, 17)
    ax.set_xlim(0, 20); ax.set_ylim(17, 0); ax.axis('off')
    for i, (lab, c) in enumerate(raters):
        yy = 2 + i * 3.4
        ax.add_patch(Rectangle((0, yy - 1.2), 3, 2.4, fc=c, ec='none'))
        ax.text(4, yy, lab, fontsize=5.8, color=INK, style='italic', va='center')
    save(fig, 'rater_legend')


# ============================================================ panel c: matrices
def export_matrices():
    fig, h = newfig(22, 30)
    h.imshow(sc.transform(X)[:32], cmap=DIV, vmin=-2.5, vmax=2.5, aspect='equal'); h.axis('off')
    save(fig, 'panel_c_matrix_coordinates')

    fig, h = newfig(28, 30)
    h.imshow(scores[:32, :23], cmap='PuOr', vmin=-5, vmax=5, aspect='equal'); h.axis('off')
    save(fig, 'panel_c_matrix_scale_pca')


# ========================= panel c: model search space -> nested-CV widget (replaces the old flat fold bar)
def export_search_and_cv():
    fig, ax = newfig(40, 12)
    ax.set_xlim(0, 40); ax.set_ylim(12, 0); ax.axis('off')
    ax.add_patch(Rectangle((0, 0), 15, 7.6, fc='#F4F6F7', ec='none', zorder=10))
    for i in range(10):
        ax.scatter([2.2 + 2.4 * (i % 5)], [2.4 if i < 5 else 5.4], s=16, c=MUTE, alpha=.85, zorder=12)
    ax.text(7.5, 9.7, '10-model search space', fontsize=5.2, color=MUTE, style='italic', ha='center')
    ax.annotate('', xy=(20, 3.8), xytext=(16, 3.8), arrowprops=dict(arrowstyle='-|>', color=MUTE, lw=1.1))
    ow = 3.2
    for i in range(4):
        ax.add_patch(Rectangle((21 + i * 4.0, 1.9), ow, 3.2, fc=INK if i == 0 else '#E2E5E8', ec='none'))
    iw = 2.2
    for i in range(3):
        ax.add_patch(Rectangle((26.8 + i * 2.7, 6.7), iw, 1.8, fc='#BFC7CD' if i == 1 else '#DEE2E5', ec='none'))
    ax.plot([25, 26.8], [5.1, 6.7], c=MUTE, lw=.7); ax.plot([36.2, 34.4], [5.1, 6.7], c=MUTE, lw=.7)
    ax.text(28.5, 9.7, '4 outer x 3 inner folds', fontsize=5.2, color=MUTE, style='italic', ha='center')
    save(fig, 'panel_c_search_space_and_cv')
    (PANELS / 'panel_c_nested_cv_fold_bar.png').unlink(missing_ok=True)  # superseded by the widget above


# ========================================== panel c: per-target model winners
def export_model_winners():
    # STN's number is its real HELD-OUT median (Figure 3, pooled across all 4
    # acquisitions), not the internal-CV screening median -- see render_figures.fig1().
    stn_med = float(data[data.method == 'AFIDs'].ED.median())
    stats = json.load(open(NEW / 'verified_summary.json'))
    winners = [('STN', CORAL, 'Ridge', stn_med),
               ('cZI', TARGET_COLORS['cZI'], stats['cZI']['model'], stats['cZI']['median']),
               ('fct', TARGET_COLORS['fct'], stats['fct']['model'], stats['fct']['median']),
               ('RN', TARGET_COLORS['RN'], stats['RN']['model'], stats['RN']['median'])]
    fig, ax = newfig(66, 32)
    ax.set_xlim(0, 66); ax.set_ylim(32, 0); ax.axis('off')
    ax.text(33, 2, 'Selected architecture per target', fontsize=5.8, color=MUTE, style='italic', ha='center')
    ax.text(33, 5.3, 'Median error -- STN: held-out (Fig. 3); others: internal CV (Fig. 4d)', fontsize=4.8, color=MUTE, style='italic', ha='center')
    for i, (t, c, model, med) in enumerate(winners):
        yy = 10 + i * 6.4
        ax.add_patch(Rectangle((0, yy - 2.6), 66, 5.2, fc='#F4F6F7', ec='none'))
        ax.add_patch(Rectangle((0, yy - 2.6), 1.3, 5.2, fc=c, ec='none'))
        ax.scatter([5], [yy], s=16, c=c); ax.text(7.5, yy, t, fontsize=6.6, weight='bold', va='center')
        ax.text(19, yy, model, fontsize=6.6, va='center')
        ax.text(48, yy, f'median {med:.2f} mm', fontsize=5.9, color=MUTE, va='center')
    save(fig, 'panel_c_model_winners')


if __name__ == '__main__':
    export_3d_scatter()
    export_lens()
    export_venn()
    export_rater_legend()
    export_matrices()
    export_search_and_cv()
    export_model_winners()
    (PANELS / 'legend_afid_and_targets.png').unlink(missing_ok=True)  # no longer part of fig1()'s design
    print('all panels exported to', PANELS)
