"""2608.12408 v3: check every number in the compiled PDF.

1. Every \\NV{key} used in the v3 source exists in results/paper_v3/numbers_manifest_v3.csv.
2. Every manifest text is a single ROUND_HALF_UP rounding of its full-precision value.
3. Every decimal number in the PDF text is a manifest value or listed in ALLOWED with the
   reason it does not change (v2-note quotations, model-vs-reference similarities that involve
   no brain data, constants, values shown in the unchanged v2 figures, the one value that
   could not be recomputed). Only the bibliography entries are excluded; appendices and figure
   pages that follow the "References" heading are checked.
4. Decimal literals typed in the v3 source body (outside \\NV) are listed.

Output: results/paper_v3/check_numbers_v3.json; exit code 1 on any failure.
"""
import json
import re
import subprocess
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAPER = REPO / "paper" / "arxiv_upload_v3"
TEX = PAPER / "Evaluation_Resolution_Confounds_paper_v3.tex"
PDF = PAPER / "Evaluation_Resolution_Confounds_paper_v3.pdf"
MAN = REPO / "results" / "paper_v3" / "numbers_manifest_v3.csv"

ALLOWED = {
    # quotations in the "Note on version 2" (historical; v2 text, kept verbatim)
    "0.001": "v2 note: table change by 0.001; also SEM / CI texts that equal it",
    "0.0013": "v2 note: precision claim quoted", "0.958": "v2 note: stability bound quoted",
    "0.957": "reference-RDM stability bound (model-free, unchanged)",
    "0.009": "v2 note: paired-SEM bound quoted", "0.007": "v2 note: paired-SEM bound quoted",
    "0.002": "v2 note: change of the paired-SEM range", "0.075": "v2 note: endpoint value quoted",
    "0.033": "v2 note: endpoint value quoted", "0.042": "v2 note: endpoint value quoted",
    "0.044": "v2 note / figure captions: v2 value in the plotted convention",
    # constants and design parameters
    "0.3": "dropout", "1.0": "gradient clip", "0.9": "SGD momentum", "0.02": "PC inference rate",
    "0.003": "STDP A+-", "0.031": "sign-flip floor 1/32", "0.017": "exact permutation p (unchanged)",
    "1.00": "rank correlation / variance ratio (unchanged)", "0.0003": "BN mean gap (no brain data)",
    "1.3": "receptive-field fraction (geometry)",
    # model-vs-reference similarities (no brain data; unchanged)
    "0.22": "ResNet Gabor similarity", "0.31": "ResNet Gabor similarity", "0.02": "CNN Gabor similarity",
    "0.03": "CNN Gabor similarity / PC alignment change 0.03-0.04", "0.04": "PC alignment change 0.03-0.04",
    "0.24": "mean-RGB similarity, untrained", "0.54": "mean-RGB similarity, backprop",
    "0.048": "colour-histogram convergence", "0.052": "colour-histogram convergence",
    "0.015": "luminance convergence, identity BN", "0.011": "luminance convergence range / variant B",
    "0.046": "luminance convergence range", "0.30": "within-filter rank correlation (unchanged)",
    "0.70": "within-filter rank correlation (unchanged)", "0.20": "within-variant correlation range",
    "0.90": "within-variant correlation range",
    # not recomputed (RDMs not stored), stated in the v2 convention
    "0.030": "leak-calibration cost (v2 convention, labelled) / v2 figure caption value",
    "0.006": "leak-calibration SEM (v2 convention, labelled)",
    # unchanged v2 figure captions (plotted convention)
    # reproducibility statements
    "2": "", "5": "",
    # Appendix B weight displacement (no brain data)
    "29.4": "weight displacement", "2.1": "weight displacement", "90.4": "weight displacement",
    "7.2": "weight displacement", "119.2": "weight displacement", "0.0": "weight displacement",
    "2.4": "initialization scale factor",
    # v3 note: v2 printed value quoted next to its recomputation
    "0.020": "v3 note: joint partial at 32 px as printed in v2",
    "0.5": "decision threshold of the criterion fixed before the upsampling analysis (R >= 0.5)",
}
# section numbers (\S3.1 ... \S4.4) and tick labels of the unchanged v2 figures
ALLOWED.update({f"{a}.{b}": "section number" for a in (3, 4) for b in range(1, 8)})
ALLOWED.update({t: "axis tick label in an unchanged v2 figure" for t in (
    "0.000", "0.00", "0.0050", "0.0075", "0.010", "0.0100", "0.0125", "0.0150", "0.0175", "0.0200", "0.0225",
    "0.01", "0.05", "0.06", "0.07", "0.08", "0.2", "0.4", "0.6", "0.8")})


def single(x, d):
    return format(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP), "f")


def norm(t):
    t = t.replace("\u2212", "-").replace("{,}", ",").lstrip("+")
    if t.startswith("-"):
        t = t[1:]
    if t.startswith("."):
        t = "0" + t
    return t


def alnum(s):
    return re.sub(r"[^0-9a-z]", "", s.lower())


def strip_bibliography(txt, tex):
    """Remove only the bibliography entries from the PDF text.

    pdftotext places the appendices and some figure pages after the "References" heading and
    interleaves wrapped entries with figure text, so the text cannot be cut at the heading.
    A line after the heading is dropped if it is a fragment of a \\bibitem entry of the source
    (compared on letters and digits only, since pdftotext drops hyphens and ligatures differ).
    """
    src = tex[tex.index("\\begin{thebibliography}"):tex.index("\\end{thebibliography}")]
    src = re.sub(r"\\bibitem\[[^\]]*\]\{[^}]*\}", " ", src)
    src = re.sub(r"\\[a-zA-Z]+", " ", src).replace("~", " ")
    bib = alnum(src.replace("fi", "").replace("fl", ""))
    lines = txt.splitlines()
    head = next(i for i, l in enumerate(lines) if l.strip() == "References")
    keep, dropped = lines[:head], []
    for l in lines[head + 1:]:
        n = alnum(l.replace("fi", "").replace("fl", "").replace("\ufb01", "").replace("\ufb02", ""))
        if len(n) >= 20 and re.search("[a-z]", n) and n in bib:
            dropped.append(l)
        else:
            keep.append(l)
    return "\n".join(keep), dropped


def main():
    fails = {}
    m = pd.read_csv(MAN, dtype={"text": str})
    tex = TEX.read_text(encoding="utf-8")
    used = set(re.findall(r"\\NV\{([^}]+)\}", tex))
    missing = sorted(used - set(m.key))
    if missing:
        fails["keys_missing"] = missing
    bad = []
    for _, r in m.iterrows():
        t = str(r.text)
        if "." in t:
            d = len(t.split(".")[1])
            s = single(r.value, d)
            if norm(s) != norm(t):
                bad.append((r.key, t, s))
    if bad:
        fails["rounding"] = bad
    allowed_vals = {norm(str(t)) for t in m.text} | set(ALLOWED)
    txt = subprocess.run(["pdftotext", "-enc", "UTF-8", str(PDF), "-"], capture_output=True).stdout.decode("utf-8")
    body, bib_lines = strip_bibliography(txt, tex)
    nums = re.findall(r"(?<![\w.])[-\u2212+]?\d*\.\d+", body)
    unknown = sorted({norm(n) for n in nums} - allowed_vals, key=float)
    if unknown:
        fails["pdf_numbers_without_source"] = unknown
    b = tex[tex.index("\\begin{document}"):tex.index("\\begin{thebibliography}")]
    b = re.sub(r"\\NV\{[^}]+\}", "", b)
    typed = sorted({norm(x) for x in re.findall(r"(?<![\w.])[-+]?\d*\.\d+", b)} - allowed_vals, key=float)
    if typed:
        fails["typed_literals_not_allowed"] = typed
    # 5. wording lint: the v2 reading of §3.5 (detail above 32 px carries the effect) is reversed
    #    in v3; the training-dynamics description is corrected; exploratory blur / band-pass
    #    results are not part of the paper.
    flat = re.sub(r"\s+", " ", tex[tex.index("\\begin{document}"):tex.index("\\begin{thebibliography}")])
    flat = flat.split("\\textbf{Note on version 3.}")[0] + flat.split("\\section{Introduction}")[1]  # version notes quote withdrawn claims
    banned = [r"carried by image detail", r"detail above [^.]{0,60}carr", r"content axis", r"requires that detail",
              r"hurts? trained ones", r"image detail and not the number", r"resize chain is introduced",
              r"property of the content and not of the pooling", r"dependence is on image content",
              r"falls epoch by epoch", r"accumulating epoch by epoch", r"\bblur", r"band-?pass", r"low-pass"]
    hits = [p for p in banned if re.search(p, flat, re.I)]
    # the upsampling analysis used a criterion fixed before the analysis but was not preregistered:
    # checked on the whole source, including the version notes
    if re.search(r"preregist", tex, re.I):
        hits.append("preregist (whole source)")
    if hits:
        fails["wording_lint"] = hits
    n_items = tex.count("\\bibitem")
    if len(bib_lines) < n_items:
        fails["bibliography_not_removed"] = f"{len(bib_lines)} lines removed for {n_items} entries"
    out = {"n_keys_used": len(used), "n_manifest": len(m), "n_pdf_decimals": len(nums),
           "n_bibliography_lines_removed": len(bib_lines), "fails": fails}
    (REPO / "results" / "paper_v3" / "check_numbers_v3.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
