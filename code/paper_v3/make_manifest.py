"""2608.12408 v3: numbers manifest and LaTeX macros.

Every number that changes from v2 to v3 gets a key; its v3 value is read from the computed
result files (never typed), rounded once (ROUND_HALF_UP) from full precision, and written as
\\NV{key} into paper/numbers_v3.tex. The manifest lists key, location, v2 printed value, v3
text, full-precision value and source.

Inputs: results/paper_v3/numbers_v3_part1.csv, numbers_v3_lowlevel.csv (compute_numbers.py,
compute_lowlevel.py), results/upsampling/step4_gaps.csv and step6_headline.csv (v12 stimulus
bootstrap), and the cited values of 2604.16875 v4 (learning-rules-rsa numbers manifest, copied
to results/paper_v3/external_2604/ with PROVENANCE).

Output: paper/numbers_v3.tex, results/paper_v3/numbers_manifest_v3.csv, NUMBERS_MANIFEST_v3.md
"""
import ast
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
P3 = REPO / "results" / "paper_v3"
UPS = REPO / "results" / "upsampling"


def rnd(x, d, sign=False):
    s = format(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP), "f")
    if s.startswith("-") and float(s) == 0:
        s = s  # keep "-0.000": v2 printed signed zeros the same way
    if sign and not s.startswith("-"):
        s = "+" + s
    return s


def parse(v):
    if isinstance(v, str) and v.strip()[:1] in "([{":
        s = re.sub(r"np\.float64\(([^)]*)\)", r"\1", v)
        return ast.literal_eval(s)
    return v


P1 = pd.read_csv(P3 / "numbers_v3_part1.csv").set_index("key")
LL = pd.read_csv(P3 / "numbers_v3_lowlevel.csv").set_index("key")
G4 = pd.read_csv(UPS / "step4_gaps.csv")
G6 = pd.read_csv(UPS / "step6_headline.csv")
EXT = pd.read_csv(P3 / "external_2604" / "numbers_manifest.csv").drop_duplicates("key").set_index("key")


def p1(key, i=None):
    v = parse(P1.loc[key, "v3_value"])
    return float(v if i is None else (v[i] if not isinstance(v, dict) else v[i]))


def ll(key, i=None):
    v = parse(LL.loc[key, "v3_value"])
    return float(v if i is None else v[i])


def g4(roi, px, col, conv="persub"):
    r = G4[(G4.convention == conv) & (G4.roi == roi) & (G4.arm == "NATIVE") & (G4.res == px)].iloc[0]
    return float(r[col])


def g6(conv, cr, px, col):
    r = G6[(G6.convention == conv) & (G6.crossrun == cr) & (G6.res == px)].iloc[0]
    return float(r[col])


def ext(key):
    return float(EXT.loc[key, "value"])


S_SWEEP = "results/paper_v3/numbers_v3_part1.csv (v12 checkpoints, per subject, cross-run)"
S_LL = "results/paper_v3/numbers_v3_lowlevel.csv (per subject, cross-run)"
S_BOOT = "results/upsampling/step4_gaps.csv (v12, stimulus bootstrap, 1000 resamples)"
S_EXT = "2604.16875 v4 numbers manifest (external_2604/)"

# key, location, v2 printed, value, decimals, signed, source
E = []


def e(key, loc, old, val, d=3, sign=False, src=S_SWEEP):
    E.append(dict(key=key, location=loc, v2_printed=old, value=val, text=rnd(val, d, sign), source=src))


def ei(key, loc, old, val, src=S_SWEEP):        # integer counts (seeds positive)
    E.append(dict(key=key, location=loc, v2_printed=old, value=val, text=str(int(val)), source=src))


# ── headline (abstract, intro, 3.1) ─────────────────────────────────────────
e("gap.V1.32", "Abstract, Intro, 3.1, Methods", "-0.001", p1("gap.V1.32"), sign=True)
e("gap.V1.32.sem", "Abstract, Intro, 3.1", "0.007", p1("gap.V1.32.sem"))
ei("gap.V1.32.pos", "3.1, Methods", "3", p1("gap.V1.32.pos"))
e("gap.V1.224", "Abstract, 3.1, Fig.1 text, 3.2, 3.4, 4.2", "+0.044", p1("gap.V1.224"), sign=True)
e("gap.V1.224.sem", "Abstract, 3.1", "0.006", p1("gap.V1.224.sem"))
ei("gap.V1.224.pos", "3.1, Methods", "5", p1("gap.V1.224.pos"))
e("boot.V1.32.lo", "Abstract", "-", g4("V1", 32, "boot_lo"), sign=False, src=S_BOOT)
e("boot.V1.32.hi", "Abstract", "-", g4("V1", 32, "boot_hi"), src=S_BOOT)
e("boot.V1.224", "Abstract", "+0.044", g4("V1", 224, "gap"), sign=True, src=S_BOOT)
e("boot.V1.224.lo", "Abstract", "-", g4("V1", 224, "boot_lo"), src=S_BOOT)
e("boot.V1.224.hi", "Abstract", "-", g4("V1", 224, "boot_hi"), src=S_BOOT)
for px in (64, 96, 128, 160):                     # all six resolutions (manifest; 96 in App. C)
    e(f"gap.V1.{px}", f"App. C ({px} px)" if px == 96 else "manifest only", "+0.025" if px == 96 else "-",
      p1(f"gap.V1.{px}"), sign=True)
    ei(f"gap.V1.{px}.pos", "manifest only", "-", p1(f"gap.V1.{px}.pos"))
# mean-RDM convention on cross-run pairs, for the Methods paragraph
e("mr.cr.V1.32", "Methods", "-0.001 (all pairs)", g6("meanrdm", True, 32, "gap"), sign=True, src="results/upsampling/step6_headline.csv")
e("mr.cr.V1.224", "Methods", "+0.044 (all pairs)", g6("meanrdm", True, 224, "gap"), sign=True, src="results/upsampling/step6_headline.csv")
e("mr.all.V1.224", "Methods, 4.2", "+0.044", g6("meanrdm", False, 224, "gap"), sign=True, src="results/upsampling/step6_headline.csv")
e("mr.all.V1.32", "Methods", "-0.001", g6("meanrdm", False, 32, "gap"), sign=True, src="results/upsampling/step6_headline.csv")

# rule levels at 32 / 224 (3.1)
for rk, (o32, o224) in {"bp": ("0.065", "0.031"), "fa": ("0.020", "0.012"), "pc": ("0.026", "0.016"),
                        "stdp": ("0.059", "0.037"), "rnd": ("0.064", "0.075")}.items():
    e(f"rho.{rk}.V1.32", "3.1", o32, p1(f"rho.{rk}.V1.32"))
    e(f"rho.{rk}.V1.224", "3.1, 3.4, Abstract (rnd)", o224, p1(f"rho.{rk}.V1.224"))

# best layer (3.1, App. C, abstract)
for px, old, oldsem in ((32, "+0.014", "0.006"), (96, "+0.041", None), (224, "+0.060", "0.004")):
    e(f"bestgap.{px}", "Abstract (32), 3.1, App. C", old, p1(f"bestgap.{px}.new", 0), sign=True)
    if oldsem:
        e(f"bestgap.{px}.sem", "3.1", oldsem, p1(f"bestgap.{px}.new", 1))
    ei(f"bestgap.{px}.pos", "3.1, App. C", {32: "4", 96: "5", 224: "5"}[px], p1(f"bestgap.{px}.new", 2))

# LOC / IT (Intro, 3.7)
e("gap.LOC.32", "Intro, 3.7", "+0.019", p1("gap.LOC.32"), sign=True)
e("gap.LOC.32.sem", "3.7", "0.001", p1("gap.LOC.32.sem"))
e("gap.LOC.224", "3.7", "+0.018", p1("gap.LOC.224"), sign=True)
e("gap.LOC.224.sem", "3.7", "0.001", p1("gap.LOC.224.sem"))
e("gap.IT.32", "3.7", "+0.015", p1("gap.IT.32"), sign=True)
e("gap.IT.32.sem", "3.7", "0.002", p1("gap.IT.32.sem"))
e("gap.IT.224", "3.7", "+0.005", p1("gap.IT.224"), sign=True)
e("gap.IT.224.sem", "3.7", "0.001", p1("gap.IT.224.sem"))
e("rho.bp.LOC.32", "3.7", "0.017", p1("rho.bp.LOC.32"))
e("rho.bp.LOC.224", "3.7", "0.013", p1("rho.bp.LOC.224"))
e("rnd.LOC.max", "3.7", "-0.002", p1("rnd.LOC.range.new", 1), sign=False)
e("rnd.LOC.min", "3.7", "-0.005", p1("rnd.LOC.range.new", 0), sign=False)

# 3.2 BN calibration
for key, old in (("cal.b-random.224", "-0.003"), ("cal.d-random.224", "-0.011"), ("cal.a-random.224", "-0.026"),
                 ("cal.c-random.224", "-0.020"), ("cal.B-A.224", "+0.023"), ("cal.D-C.224", "+0.008"),
                 ("cal.D-B.224", "-0.008"), ("cal.b-bp.224", "+0.041"), ("cal.d-bp.224", "+0.033")):
    e(key, "3.2", old, p1(key + ".new", 0), sign=True, src=S_SWEEP.replace("v12 checkpoints", "bncal RDMs, BP v12"))
    e(key + ".sem", "3.2", "-", p1(key + ".new", 1), src="bncal RDMs")
    ei(key + ".pos", "3.2", "-", p1(key + ".new", 2), src="bncal RDMs")
sems = [p1(f"cal.{v}-bp.224.new", 1) for v in "abcd"]
e("cal.sem.min", "3.2", "0.001", min(sems), src="bncal RDMs (variant - BP @224, SEM)")
e("cal.sem.max", "3.2", "0.007", max(sems), src="bncal RDMs (variant - BP @224, SEM)")

# 3.3 architectures (V1 at 32 and 224 px)
for m, (o32, o224) in {"resnet50": ("0.045", "0.032"), "swin_t": ("0.079", "0.052")}.items():
    d = parse(P1.loc[f"arch.{m}.V1.new", "v3_value"])
    e(f"arch.{m}.32", "3.3", o32, float(d[32]), src="data/arch_rdms (extract_arch.py), per subject, cross-run")
    e(f"arch.{m}.224", "3.3", o224, float(d[224]), src="data/arch_rdms (extract_arch.py), per subject, cross-run")

# 3.4 low-level
e("ref.lum.V1", "Abstract, Methods, Intro, 3.4", "0.074", ll("ref.MEANLUM.V1"), src=S_LL)
e("ref.hist.V1", "3.4", "0.045", ll("ref.COLORHIST.V1"), src=S_LL)
e("ref.rgb.V1", "3.4", "0.028", ll("ref.MEANRGB.V1"), src=S_LL)
e("ref.pixel.V1", "3.4", "0.030", ll("ref.PIXEL.V1"), src=S_LL)
e("gabor.V1.min", "3.4", "0.018", ll("gabor.V1.range", 0), src=S_LL)
e("gabor.V1.max", "3.4", "0.037", ll("gabor.V1.range", 1), src=S_LL)
e("unt.partial_lum", "3.4", "0.038", ll("unt.partial_lum.224.new"), src=S_LL)
e("unt.pixel_decrease", "3.4", "0.004", ll("unt.pixel_decrease.224.new"), src=S_LL)
e("gap.joint4.32", "3.4", "-0.020", ll("gap.joint4.32.new", 0), sign=True, src=S_LL)
e("gap.joint4.224", "3.4", "+0.036", ll("gap.joint4.224.new", 0), sign=True, src=S_LL)
e("gap.lum.224", "3.4", "+0.026", ll("gap.lum.224.new", 0), sign=True, src=S_LL)
e("six.order", "3.4", "0.94", ll("six.order.new"), d=2, src=S_LL)
e("within.rho25", "3.4", "0.87", ll("within.lum.rho25.new"), d=2, src=S_LL)
e("within.B.dV1", "3.4", "+0.011", ll("within.B.dV1.new", 0), sign=True, src=S_LL)
ei("within.B.dV1.pos", "3.4", "4", ll("within.B.dV1.new", 2), src=S_LL)

# 3.5 content control (v12 NATIVE / UPSAMPLED)
e("ups.step.bp", "3.5", "-0.033", p1("ups.step.bp.new", 0), sign=True)
e("ups.step.rnd", "3.5", "-0.000", p1("ups.step.rnd.new", 0), sign=True)
for arm, o, osem in (("NATIVE", "+0.030", "0.002"), ("UPSAMPLED", "+0.003", "0.001")):
    e(f"ups.gapopen.{arm}", "Abstract, 3.5, 4", o, p1(f"ups.gapopen.{arm}.new", 0), sign=True)
    e(f"ups.gapopen.{arm}.sem", "Abstract, 3.5", osem, p1(f"ups.gapopen.{arm}.new", 1))
    ei(f"ups.gapopen.{arm}.pos", "3.5", "5" if arm == "NATIVE" else "-", p1(f"ups.gapopen.{arm}.new", 2))
e("ups.bp.NATIVE", "Abstract, 3.5", "-0.023", p1("ups.slope.bp.NATIVE.new", 0), sign=True)
e("ups.bp.NATIVE.sem", "3.5", "0.002", p1("ups.slope.bp.NATIVE.new", 1))
e("ups.bp.UPSAMPLED", "Abstract, 3.5", "-0.000", p1("ups.slope.bp.UPSAMPLED.new", 0), sign=True)
e("ups.bp.UPSAMPLED.sem", "3.5", "0.001", p1("ups.slope.bp.UPSAMPLED.new", 1))
ei("ups.bp.UPSAMPLED.pos", "Abstract, 3.5", "2", p1("ups.slope.bp.UPSAMPLED.new", 2))
for rk, o_n, o_u in (("fa", "-0.004", "+0.003"), ("pc", "-0.005", "+0.004")):
    e(f"ups.{rk}.NATIVE", "3.5", o_n, p1(f"ups.slope.{rk}.NATIVE.new", 0), sign=True)
    e(f"ups.{rk}.UPSAMPLED", "3.5", o_u, p1(f"ups.slope.{rk}.UPSAMPLED.new", 0), sign=True)
e("ups.rnd.UPSAMPLED", "3.5", "+0.0029", p1("ups.slope.rnd.UPSAMPLED.new", 0), d=4, sign=True)
e("ups.rnd.UPSAMPLED.sem", "3.5", "0.0003", p1("ups.slope.rnd.UPSAMPLED.new", 1), d=4)
e("ups.rnd.frac", "3.5", "44", 100 * p1("ups.slope.rnd.UPSAMPLED.new", 0) / p1("ups.slope.rnd.NATIVE.new", 0), d=0)
e("ups.removed", "4, Conclusion", "90", 100 * (1 - p1("ups.gapopen.UPSAMPLED.new", 0) / p1("ups.gapopen.NATIVE.new", 0)), d=0)

# 3.6 training dynamics (per subject; now on cross-run pairs)
S_TD = "training-dynamics RDMs (Modal), per subject, cross-run"
e("td.bp.224", "3.6", "-0.031", p1("td.backprop.224.new", 0), sign=True, src=S_TD)
e("td.bp.224.sem", "3.6", "0.005", p1("td.backprop.224.new", 1), src=S_TD)
ei("td.bp.224.neg", "3.6", "5", p1("td.backprop.224.new", 2), src=S_TD)
e("td.bp.32", "3.6", "-0.000", p1("td.backprop.32.new", 0), sign=True, src=S_TD)
e("td.bp.32.sem", "3.6", "0.005", p1("td.backprop.32.new", 1), src=S_TD)
ei("td.bp.32.neg", "3.6", "3", p1("td.backprop.32.new", 2), src=S_TD)
e("td.pc.32", "3.6", "-0.026", p1("td.predictive_coding.32.new", 0), sign=True, src=S_TD)
e("td.pc.32.sem", "3.6", "0.004", p1("td.predictive_coding.32.new", 1), src=S_TD)
e("td.fa.32", "3.6", "-0.021", p1("td.feedback_alignment.32.new", 0), sign=True, src=S_TD)
e("td.fa.32.sem", "3.6", "0.007", p1("td.feedback_alignment.32.new", 1), src=S_TD)

# cited from 2604.16875 v4 (endpoint study) and its lower bound
e("p1.rnd", "Intro", "0.075", ext("rho.ps.cr.bn.rnd.V1"), src=S_EXT)
e("p1.bp", "Intro", "0.033", ext("rho.ps.cr.bn.bp.V1"), src=S_EXT)
e("p1.gap", "Intro, 4.2", "+0.042", ext("d.ps.cr.bn.rnd.bp.V1"), sign=True, src=S_EXT)
e("p1.gap.lo", "Intro", "-", ext("d.ps.cr.bn.rnd.bp.V1.lo"), src=S_EXT)
e("p1.gap.hi", "Intro", "-", ext("d.ps.cr.bn.rnd.bp.V1.hi"), src=S_EXT)
e("p1.pr.min", "3.4", "0.004", -ext("pr.d.V1.max"), src=S_EXT + " (partial-RSA decrease, V1)")
e("p1.pr.max", "3.4", "0.008", -ext("pr.d.V1.min"), src=S_EXT + " (partial-RSA decrease, V1)")
for roi in ("V1", "V2", "LOC", "IT"):
    e(f"lb.{roi}", "Methods (scale)", "-", ext(f"lb.cr.{roi}"), src=S_EXT + " (Nili LOSO lower bound, cross-run)")
e("lb.V1.lo", "Methods (scale)", "-", ext("lb.cr.V1.lo"), src=S_EXT)
e("lb.V1.hi", "Methods (scale)", "-", ext("lb.cr.V1.hi"), src=S_EXT)


# ── 3.5 rewrite: upsampling test (step4, per subject, cross-run, v12, bootstrap CI) ──
S4R = "results/upsampling/step4_ratio.csv (v12, per subject, cross-run, 1000 stimulus resamples)"
R4 = pd.read_csv(UPS / "step4_ratio.csv")
R4 = R4[R4.convention == "persub"].set_index("roi")
for roi in ("V1", "LOC"):
    r = R4.loc[roi]
    e(f"s4.{roi}.up224", "3.5, Abstract", "-", r.gap_up_224, sign=True, src=S4R)
    e(f"s4.{roi}.up224.lo", "3.5, Abstract", "-", r.gap_up_lo, src=S4R)
    e(f"s4.{roi}.up224.hi", "3.5, Abstract", "-", r.gap_up_hi, src=S4R)
    e(f"s4.{roi}.R", "3.5, Abstract, note", "-", r.R, d=2, src=S4R)
    e(f"s4.{roi}.R.lo", "3.5, Abstract, note", "-", r.R_lo, d=2, src=S4R)
    e(f"s4.{roi}.R.hi", "3.5, Abstract, note", "-", r.R_hi, d=2, src=S4R)
    e(f"s4.{roi}.diff", "3.5", "-", r.up_minus_nat, sign=True, src=S4R)
    e(f"s4.{roi}.diff.lo", "3.5", "-", r.up_minus_nat_lo, src=S4R)
    e(f"s4.{roi}.diff.hi", "3.5", "-", r.up_minus_nat_hi, src=S4R)
e("s4.V1.up64", "3.5", "-", g4up := float(G4[(G4.convention == "persub") & (G4.roi == "V1") & (G4.arm == "UPSAMPLED")
                                             & (G4.res == 64)].gap.iloc[0]), sign=True, src="results/upsampling/step4_gaps.csv")
_rs = pd.concat([pd.read_csv(UPS / f"rsa_seed{i}.csv") for i in range(5)])
_rs = _rs[(_rs.variant == "cr") & (_rs.convention == "persub") & (_rs.layer == "Conv1") & (_rs.roi == "V1")]
_bpu = _rs[(_rs.rule == "Backprop") & (_rs.arm == "UPSAMPLED") & (_rs.res >= 64)].groupby("res").rho.mean()
e("s4.bp.up.min", "3.5", "-", _bpu.min(), src="results/upsampling/rsa_seed*.csv")
e("s4.bp.up.max", "3.5", "-", _bpu.max(), src="results/upsampling/rsa_seed*.csv")
e("s4.bp.up.64", "3.5", "-", _bpu.loc[64], src="results/upsampling/rsa_seed*.csv")

# ── 3.6 training dynamics: backprop at 224 px, epoch 0 -> 1 -> 40 (seed mean, per subject, cross-run) ──
_FD = pd.read_csv(P3 / "figure_data_v3.csv").set_index("cell")
for ep in (0, 1, 40):
    e(f"td.bp224.e{ep}", "3.6", "-", float(_FD.loc[f"td|Backprop|224|{ep}", "mean"]),
      src="results/paper_v3/figure_data_v3.csv (training-dynamics RDMs)")


def main():
    m = pd.DataFrame(E)
    assert m.key.is_unique, m[m.key.duplicated()]
    m.to_csv(P3 / "numbers_manifest_v3.csv", index=False)
    lines = ["% generated by code/paper_v3/make_manifest.py -- do not edit", "\\makeatletter"]
    for _, r in m.iterrows():
        lines.append(f"\\expandafter\\def\\csname nv@{r.key}\\endcsname{{{r.text}}}")
    lines += ["\\makeatother",
              "\\newcommand{\\NV}[1]{\\ifcsname nv@#1\\endcsname\\csname nv@#1\\endcsname"
              "\\else\\errmessage{Undefined number: #1}\\fi}"]
    (REPO / "paper" / "arxiv_upload_v3" / "numbers_v3.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    md = ["# Numbers manifest, 2608.12408 v2 -> v3", "",
          "Primary convention of v3: per subject, cross-run stimulus pairs (210,205 of 258,840). "
          "v2 printed values used the mean-RDM convention on all pairs (training dynamics: per subject, "
          "all pairs). Generated by `code/paper_v3/make_manifest.py`.", "",
          "| key | location | v2 printed | v3 | source |", "|---|---|---|---|---|"]
    md += [f"| `{r.key}` | {r.location} | {r.v2_printed} | {r.text} | {r.source} |" for _, r in m.iterrows()]
    (P3 / "NUMBERS_MANIFEST_v3.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    pd.set_option("display.width", 200, "display.max_rows", 300, "display.max_colwidth", 40)
    print(m[["key", "v2_printed", "text"]].to_string(index=False))


if __name__ == "__main__":
    main()
