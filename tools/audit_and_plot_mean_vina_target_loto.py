import os
import argparse
import uuid
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Public default: resources beside this source; historical resources are explicit overrides.
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "runs" / ("statistics-" + uuid.uuid4().hex)
PICTURE_DIR = OUTPUT_ROOT / "picture"
PER_TARGET_CSV = None
DETAILED_CSV = None

def configure_paths(argv=None):
    global PROJECT_ROOT, OUTPUT_ROOT, PICTURE_DIR, PER_TARGET_CSV, DETAILED_CSV
    parser = argparse.ArgumentParser(description="Analyze explicit docking result tables")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--per-target-csv", type=Path, required=True)
    parser.add_argument("--detailed-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    PROJECT_ROOT = args.project_root.expanduser().resolve()
    OUTPUT_ROOT = args.output_dir.expanduser().resolve()
    PER_TARGET_CSV = args.per_target_csv.expanduser().resolve()
    DETAILED_CSV = args.detailed_csv.expanduser().resolve()
    for input_file in (PER_TARGET_CSV, DETAILED_CSV):
        if not input_file.is_file():
            parser.error(f"Input file does not exist: {input_file}")
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        parser.error("Output directory is nonempty; choose a new --output-dir")
    PICTURE_DIR = OUTPUT_ROOT / "picture"
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    return args

# Aesthetic configuration
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

COLOR_POS = '#2E7D32'   # Forest green for positive Delta (favoring Ki-E2PO)
COLOR_NEG = '#C62828'   # Deep red for negative Delta (favoring TG-MolDiff)
COLOR_ZERO = '#7A7A7A'  # Medium gray for y=0 baseline
COLOR_FULL = '#2C3E50'  # Slate blue for Full-12 reference
COLOR_MAIN = '#1F77B4'  # Standard blue for LOTO points

targets = ['ESR1', 'HCRTR1', 'JAK1', 'P2RX3', 'KDM1A', 'IDH1', 'RIOK1', 'NR4A1', 'GRIK1', 'CCR9', 'FTO', 'SPIN1']

def reaudit_docking_data():
    df_per = pd.read_csv(PER_TARGET_CSV)
    df_det = pd.read_csv(DETAILED_CSV)
    per_cols = ['Model', 'Target', 'Total_Generated', 'Docking_Success_Rate', 'Mean_Vina']
    det_cols = ['Model', 'Target', 'vina_score']
    for df, cols in [(df_per, per_cols), (df_det, det_cols)]:
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError('Missing required columns: ' + ', '.join(sorted(missing)))
        if df[cols].isna().any().any():
            raise ValueError('Missing values are not allowed')
    per_models = ['PathB-50k-NoClamp', 'Ki-E2PO-v3-15k']
    det_models = ['Base TG-MolDiff', 'Ki-E2PO-v3-15k']
    expected = {(m,t) for m in per_models for t in targets}
    actual = list(zip(df_per.Model, df_per.Target))
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Expected exactly one summary per model and each of the fixed 12 targets')
    if set(zip(df_det.Model, df_det.Target)) != {(m,t) for m in det_models for t in targets}:
        raise ValueError('Detailed table must cover both models and the same 12 targets, without unknown groups')
    for col in ['Total_Generated', 'Docking_Success_Rate', 'Mean_Vina']:
        df_per[col] = pd.to_numeric(df_per[col], errors='raise')
        if not np.isfinite(df_per[col]).all():
            raise ValueError('Non-finite summary values')
    df_det['vina_score'] = pd.to_numeric(df_det['vina_score'], errors='raise')
    if not np.isfinite(df_det.vina_score).all():
        raise ValueError('Non-finite detailed scores')
    if ((df_per.Total_Generated <= 0) | (df_per.Total_Generated % 1 != 0)).any():
        raise ValueError('Total_Generated must be positive integers')
    if not df_per.Docking_Success_Rate.between(0, 1).all():
        raise ValueError('Docking_Success_Rate outside [0, 1]')
    records=[]
    for t in targets:
        row={'Target':t, 'Same_Protocol':'UNVERIFIED', 'Status':'INPUT_VALIDATED_PROTOCOL_UNVERIFIED'}
        for pm, dm, prefix in zip(per_models, det_models, ['TG_MolDiff', 'Ki_E2PO']):
            summary=df_per[(df_per.Model==pm)&(df_per.Target==t)].iloc[0]
            detail=df_det[(df_det.Model==dm)&(df_det.Target==t)]
            valid=int((detail.vina_score < 0).sum())
            if len(detail)>summary.Total_Generated:
                raise ValueError('Detailed records exceed total generated')
            # Retain historical primary summary Mean_Vina and negative-score count convention.
            # Detailed data may be a subset: no claim of verified denominator or protocol alignment.
            row[prefix+'_Total_Gen']=summary.Total_Generated
            row[prefix+'_Valid_Docked']=valid
            row[prefix+'_Mean_Vina']=summary.Mean_Vina
        delta=row['TG_MolDiff_Mean_Vina']-row['Ki_E2PO_Mean_Vina']
        row['Delta_Mean_Vina']=delta
        row['Favors_Model']='Ki-E2PO' if delta>0 else 'TG-MolDiff' if delta<0 else 'Tie'
        records.append(row)
    df_audit=pd.DataFrame(records)
    for out in [OUTPUT_ROOT, PICTURE_DIR]:
        df_audit.to_csv(out/'mean_vina_target_audit.csv',index=False)
    return df_audit

def paired_mean_t_interval(values):
    mean = float(np.mean(values))
    se = float(stats.sem(values))
    return (mean, mean) if se == 0 else stats.t.interval(0.95, df=len(values)-1, loc=mean, scale=se)

def run_statistical_and_loto_audit(df_audit):
    print("\n=== Phase 2: Running Statistical & LOTO Audits ===")
    
    diff_vina = df_audit.set_index('Target')['Delta_Mean_Vina']
    
    # 1. Full-12 Macro Delta
    full_macro_delta = diff_vina.mean()
    
    # 2. Target-level Bootstrap (n=10,000)
    np.random.seed(42)
    def calc_target_boot_ci(data_arr, n_boot=10000, ci=95):
        boot_means = [np.mean(np.random.choice(data_arr, size=len(data_arr), replace=True)) for _ in range(n_boot)]
        return np.percentile(boot_means, (100 - ci) / 2.0), np.percentile(boot_means, 100 - (100 - ci) / 2.0)
        
    full_boot_low, full_boot_high = calc_target_boot_ci(diff_vina.values)
    full_t_low, full_t_high = paired_mean_t_interval(diff_vina.values)
    
    # 3. LOTO Calculations
    loto_records = []
    raw_pvals = []
    
    for excl_t in targets:
        sub_diff = diff_vina.drop(excl_t).values
        sub_macro_delta = np.mean(sub_diff)
        b_low, b_high = calc_target_boot_ci(sub_diff)
        t_low, t_high = paired_mean_t_interval(sub_diff)
        
        w_stat, p_val = (0.0, 1.0) if np.all(sub_diff == 0) else stats.wilcoxon(sub_diff, alternative='greater')
        raw_pvals.append(p_val)
        
        loto_records.append({
            'Excluded_Target': excl_t,
            'LOTO_Macro_Delta': sub_macro_delta,
            'Delta_Shift_from_Full': sub_macro_delta - full_macro_delta,
            'Abs_Influence': abs(sub_macro_delta - full_macro_delta),
            'CI_Boot_Low': b_low,
            'CI_Boot_High': b_high,
            'CI_t_Low': t_low,
            'CI_t_High': t_high,
            'Wilcoxon_Raw_P': p_val
        })
        
    df_loto = pd.DataFrame(loto_records)
    
    # BH Correction across 12 LOTO raw P-values
    rejected, q_values, _, _ = multipletests(raw_pvals, alpha=0.05, method='fdr_bh')
    df_loto['BH_Adjusted_q'] = q_values
    df_loto['Significant_q_lt_005'] = rejected
    
    # Save BH Audit CSV
    bh_csv = PICTURE_DIR / "loto_bh_audit.csv"
    df_loto[['Excluded_Target', 'Wilcoxon_Raw_P', 'BH_Adjusted_q', 'Significant_q_lt_005', 'LOTO_Macro_Delta', 'CI_Boot_Low', 'CI_Boot_High']].to_csv(bh_csv, index=False)
    df_loto[['Excluded_Target', 'Wilcoxon_Raw_P', 'BH_Adjusted_q', 'Significant_q_lt_005', 'LOTO_Macro_Delta', 'CI_Boot_Low', 'CI_Boot_High']].to_csv(OUTPUT_ROOT / "loto_bh_audit.csv", index=False)
    print(f"Saved BH Audit CSV -> {bh_csv}")
    
    # Save Bootstrap Audit Text Report
    txt_lines = []
    txt_lines.append("=== Audited Target-Level Bootstrap & LOTO Report ===")
    txt_lines.append(f"Full-12 Target-Level Macro Delta: {full_macro_delta:+.6f} kcal/mol")
    txt_lines.append(f"Full-12 Target-Level Bootstrap 95% CI (n=10,000): [{full_boot_low:.4f}, {full_boot_high:.4f}]")
    txt_lines.append(f"Full-12 Target-Level Paired-t 95% CI: [{full_t_low:.4f}, {full_t_high:.4f}]")
    txt_lines.append("\n--- 12 Leave-One-Target-Out (LOTO) Iterations ---")
    for _, r in df_loto.iterrows():
        t = r['Excluded_Target']
        txt_lines.append(f"Excl. {t:8s}: Sub_Macro_Delta = {r['LOTO_Macro_Delta']:+.4f} kcal/mol | Shift = {r['Delta_Shift_from_Full']:+.4f} | Boot 95% CI = [{r['CI_Boot_Low']:.4f}, {r['CI_Boot_High']:.4f}] | BH q = {r['BH_Adjusted_q']:.4f}")

    txt_content = "\n".join(txt_lines)
    txt_out = PICTURE_DIR / "loto_bootstrap_audit.txt"
    txt_out.write_text(txt_content)
    (OUTPUT_ROOT / "loto_bootstrap_audit.txt").write_text(txt_content)
    print(f"Saved Bootstrap Audit Text File -> {txt_out}")
    
    return df_loto, full_macro_delta, full_boot_low, full_boot_high

def plot_audited_figure(df_audit, df_loto, full_macro_delta, full_boot_low, full_boot_high):
    print("\n=== Phase 3: Plotting 2-Panel Audited Figure ===")
    
    diff_vina = df_audit.set_index('Target')['Delta_Mean_Vina']
    
    # Dynamic calculations
    n_favor = (diff_vina > 0).sum()
    n_against = (diff_vina < 0).sum()
    
    max_pos_target = diff_vina.idxmax()
    max_pos_val = diff_vina.max()
    
    max_neg_target = diff_vina.idxmin()
    max_neg_val = diff_vina.min()
    
    # Automatically detect most influential target in LOTO
    most_influential_row = df_loto.loc[df_loto['Abs_Influence'].idxmax()]
    inf_target = most_influential_row['Excluded_Target']
    inf_delta = most_influential_row['LOTO_Macro_Delta']
    inf_ci_low = most_influential_row['CI_Boot_Low']
    inf_ci_high = most_influential_row['CI_Boot_High']
    inf_q = most_influential_row['BH_Adjusted_q']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), dpi=600)
    fig.subplots_adjust(wspace=0.25, bottom=0.15)
    
    # -------------------------------------------------------------
    # Panel (a): Target-wise Mean Vina Difference
    # -------------------------------------------------------------
    ax_a = axes[0]
    x = np.arange(len(targets))
    vals = diff_vina.values
    
    ax_a.axhline(0, color=COLOR_ZERO, linestyle='--', linewidth=1.1, zorder=1)
    
    for i, val in enumerate(vals):
        c = COLOR_POS if val >= 0 else COLOR_NEG
        ax_a.plot([i, i], [0, val], color=c, linewidth=1.8, zorder=2)
        ax_a.scatter(i, val, color=c, s=60, zorder=3, edgecolors='white', linewidths=0.8)

    # Dynamic annotation for max positive target
    max_pos_idx = targets.index(max_pos_target)
    ax_a.annotate(f'{max_pos_target} ({max_pos_val:+.2f})', xy=(max_pos_idx, max_pos_val), xytext=(max_pos_idx, max_pos_val - 0.45),
                arrowprops=dict(arrowstyle="->", color=COLOR_POS, lw=1.2),
                fontsize=8.5, fontweight='bold', color=COLOR_POS, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor=COLOR_POS, linewidth=0.8))
    
    # Dynamic annotation for max negative target
    max_neg_idx = targets.index(max_neg_target)
    ax_a.annotate(f'{max_neg_target} ({max_neg_val:.2f})', xy=(max_neg_idx, max_neg_val), xytext=(max_neg_idx - 0.55, max_neg_val + 0.65),
                arrowprops=dict(arrowstyle="->", color=COLOR_NEG, lw=1.2),
                fontsize=8.5, fontweight='bold', color=COLOR_NEG, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor=COLOR_NEG, linewidth=0.8))

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(targets, rotation=35, ha='right', fontsize=10, fontweight='medium')
    ax_a.set_ylabel("Δ Mean Vina (TG-MolDiff − Ki-E2PO, kcal/mol)", fontsize=10.5, fontweight='bold')
    ax_a.set_title("a  Target-wise Mean Vina Difference", fontsize=12, fontweight='bold', loc='left', pad=12)
    ax_a.grid(axis='y', linestyle=':', color='#DDDDDD', alpha=0.7)
    
    # Dynamic banner text (X/12 targets favor Ki-E2PO)
    banner_text = f"{n_favor}/12 Targets Favor Ki-E2PO"
    ax_a.text(0.98, 0.94, banner_text, transform=ax_a.transAxes, fontsize=9.5, fontweight='bold',
              ha='right', va='top', color=COLOR_POS,
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', edgecolor=COLOR_POS, linewidth=1.0))

    # -------------------------------------------------------------
    # Panel (b): Leave-One-Target-Out (LOTO) Sensitivity
    # -------------------------------------------------------------
    ax_b = axes[1]
    
    means = df_loto['LOTO_Macro_Delta'].values
    ci_lows = df_loto['CI_Boot_Low'].values
    ci_highs = df_loto['CI_Boot_High'].values
    yerr_lower = np.maximum(0, means - ci_lows)
    yerr_upper = np.maximum(0, ci_highs - means)
    
    ax_b.axhline(0, color=COLOR_ZERO, linestyle='--', linewidth=1.1, label='Zero Difference (y = 0)', zorder=1)
    ax_b.axhline(full_macro_delta, color=COLOR_FULL, linestyle=':', linewidth=1.8, label=f'Full-12 Difference = {full_macro_delta:+.4f} kcal/mol', zorder=2)
    ax_b.axhspan(full_boot_low, full_boot_high, color=COLOR_FULL, alpha=0.08, label='Full-12 95% CI Band', zorder=1)
    
    for i, t in enumerate(targets):
        mean_val = means[i]
        err_low = yerr_lower[i]
        err_high = yerr_upper[i]
        
        if t == inf_target:
            # Highlight most influential target
            ax_b.errorbar(i, mean_val, yerr=[[err_low], [err_high]], fmt='o', color=COLOR_NEG,
                        ecolor=COLOR_NEG, elinewidth=2.2, capsize=5, capthick=1.8, markersize=8, zorder=5)
            
            box_text = f"Excluding {inf_target} (Highest Influence)\nΔ = {inf_delta:+.4f} kcal/mol\n95% CI: [{inf_ci_low:.3f}, {inf_ci_high:.3f}]\nBH-adjusted q = {inf_q:.4f}"
            ax_b.annotate(box_text,
                        xy=(i, mean_val), xytext=(i - 1.25, mean_val + 0.35),
                        arrowprops=dict(arrowstyle="->", color=COLOR_NEG, lw=1.4),
                        fontsize=8.5, fontweight='medium', color='#111111', ha='center',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8F8', edgecolor=COLOR_NEG, linewidth=1.2))
        else:
            ax_b.errorbar(i, mean_val, yerr=[[err_low], [err_high]], fmt='o', color=COLOR_MAIN,
                        ecolor=COLOR_MAIN, elinewidth=1.4, capsize=4, capthick=1.2, markersize=6.5, zorder=4)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"Excl. {t}" for t in targets], rotation=35, ha='right', fontsize=10, fontweight='medium')
    
    for tick_label, t in zip(ax_b.get_xticklabels(), targets):
        if t == inf_target:
            tick_label.set_fontweight('bold')
            tick_label.set_color(COLOR_NEG)
            
    ax_b.set_ylabel("Δ Mean Vina (TG-MolDiff − Ki-E2PO, kcal/mol)", fontsize=10.5, fontweight='bold')
    ax_b.set_title("b  Leave-One-Target-Out (LOTO) Sensitivity", fontsize=12, fontweight='bold', loc='left', pad=12)
    # Autoscale to the actual means and confidence intervals; do not hide synthetic/extreme inputs.
    ax_b.grid(axis='y', linestyle=':', color='#DDDDDD', alpha=0.7)
    ax_b.legend(frameon=True, facecolor='white', edgecolor='#CCCCCC', fontsize=8.5, loc='lower left')
    
    # Save Figure outputs
    out_png = PICTURE_DIR / "figure_mean_vina_target_loto_audited.png"
    out_pdf = PICTURE_DIR / "figure_mean_vina_target_loto_audited.pdf"
    out_svg = PICTURE_DIR / "figure_mean_vina_target_loto_audited.svg"
    
    fig.savefig(out_png, dpi=600, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_svg, bbox_inches='tight')
    
    fig.savefig(OUTPUT_ROOT / "figure_mean_vina_target_loto_audited.png", dpi=600, bbox_inches='tight')
    fig.savefig(OUTPUT_ROOT / "figure_mean_vina_target_loto_audited.pdf", bbox_inches='tight')
    fig.savefig(OUTPUT_ROOT / "figure_mean_vina_target_loto_audited.svg", bbox_inches='tight')
    
    plt.close()
    print("\n=== Audited Figure Generated Successfully! ===")
    print(f"  PNG (600 dpi): {out_png}")
    print(f"  PDF: {out_pdf}")
    print(f"  SVG: {out_svg}")
    
    return n_favor, n_against, max_pos_target, max_neg_target, inf_target

def main(argv=None):
    configure_paths(argv)
    try:
        df_audit = reaudit_docking_data()
        df_loto, mean, low, high = run_statistical_and_loto_audit(df_audit)
        plot_audited_figure(df_audit, df_loto, mean, low, high)
    except (ValueError, KeyError, OSError, pd.errors.ParserError) as exc:
        print('Input/analysis error: ' + str(exc), file=sys.stderr)
        return 2
    positive = int((df_loto.LOTO_Macro_Delta > 0).sum())
    significant = int(df_loto.Significant_q_lt_005.sum())
    print('Protocol alignment: UNVERIFIED (scores alone cannot prove matching protocols)')
    print(f'Macro difference: {mean:+.6f}; bootstrap 95% CI [{low:.6f}, {high:.6f}]')
    print(f'LOTO positive: {positive}/{len(df_loto)}; BH-significant: {significant}/{len(df_loto)}')
    print(f'BH q range: [{df_loto.BH_Adjusted_q.min():.6g}, {df_loto.BH_Adjusted_q.max():.6g}]')
    print('LOTO family: one-sided Wilcoxon greater; BH alpha=0.05 across the 12 exclusions.')
    print('No BH-significant positive LOTO comparison.' if significant == 0 else
          'Some LOTO comparisons are significant; this is not proof of universal improvement or causal benefit.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
