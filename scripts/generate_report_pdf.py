"""
Génère reports/rapport_aps_scania.pdf depuis reports/rapport_aps_scania.md
Usage: python scripts/generate_report_pdf.py
"""
from __future__ import annotations

import os
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "reports" / "rapport_aps_scania.md"
FIG_DIR = ROOT / "reports" / "figures"
PDF_PATH = ROOT / "reports" / "rapport_aps_scania.pdf"


def _ensure_fpdf():
    try:
        from fpdf import FPDF  # noqa: F401
    except ImportError:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2", "-q"])


def _default_metrics() -> dict:
    return {
        "pos_pct": "1.67",
        "n_pos": "1000",
        "n_neg": "59000",
        "train_shape": "(60000, 171)",
        "test_shape": "(16000, 171)",
        "scale_pos_weight": "59.0",
        "mean_missing_pct": "—",
        "features_missing_gt50pct": "—",
        "vif_count_gt5": "—",
        "imbalance_A_f1_mean": "—",
        "imbalance_A_f1_std": "—",
        "imbalance_A_mcc_mean": "—",
        "imbalance_A_mcc_std": "—",
        "imbalance_B_f1_mean": "—",
        "imbalance_B_f1_std": "—",
        "imbalance_B_mcc_mean": "—",
        "imbalance_B_mcc_std": "—",
        "vif_top10_features": "voir notebook 01",
    }


def _load_metrics() -> dict:
    m = _default_metrics()
    jp = ROOT / "reports" / "metrics_summary.json"
    if jp.exists():
        import json

        with open(jp, encoding="utf-8") as f:
            raw = json.load(f)
        flat = {}

        def flatten(prefix, obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    flatten(f"{prefix}_{k}" if prefix else k, v)
            else:
                flat[prefix] = obj

        flatten("", raw)
        for k, v in flat.items():
            if isinstance(v, float):
                m[k] = f"{v:.4f}"
            elif isinstance(v, list):
                m[k] = ", ".join(str(x) for x in v[:10])
            elif not isinstance(v, dict):
                m[k] = str(v)

    try:
        sys.path.insert(0, str(ROOT))
        from src.data.loader import load_aps_data, get_X_y

        train_df, test_df = load_aps_data(project_root=str(ROOT))
        X, y = get_X_y(train_df)
        m["train_shape"] = str(tuple(train_df.shape))
        m["test_shape"] = str(tuple(test_df.shape))
        m["n_pos"] = str(int(y.sum()))
        m["n_neg"] = str(int((y == 0).sum()))
        m["pos_pct"] = f"{y.mean() * 100:.2f}"
        m["scale_pos_weight"] = f"{(y == 0).sum() / max(y.sum(), 1):.1f}"
        miss = X.isna().mean()
        m["mean_missing_pct"] = f"{miss.mean() * 100:.1f}"
        m["features_missing_gt50pct"] = str(int((miss > 0.5).sum()))
    except Exception:
        pass

    proc = ROOT / "data" / "processed" / "eda_results.joblib"
    if proc.exists():
        import joblib
        import numpy as np

        eda = joblib.load(proc)
        comp = eda.get("comparison_scores", {})
        for name, key in [("class_weight", "A"), ("smote", "B")]:
            s = comp.get(name, {})
            f1 = np.array(s.get("f1_macro", []))
            mcc = np.array(s.get("mcc", []))
            if len(f1):
                m[f"imbalance_{key}_f1_mean"] = f"{f1.mean():.4f}"
                m[f"imbalance_{key}_f1_std"] = f"{f1.std():.4f}"
            if len(mcc):
                m[f"imbalance_{key}_mcc_mean"] = f"{mcc.mean():.4f}"
                m[f"imbalance_{key}_mcc_std"] = f"{mcc.std():.4f}"
        vif_df = eda.get("vif_df")
        if vif_df is not None:
            m["vif_count_gt5"] = str(int((vif_df["VIF"] > 5).sum()))
            m["vif_top10_features"] = ", ".join(
                vif_df.sort_values("VIF", ascending=False).head(10)["Feature"].astype(str)
            )
    return m


def _fill_placeholders(text: str, metrics: dict) -> str:
    def repl(match):
        key = match.group(1).strip()
        return str(metrics.get(key, match.group(0)))

    text = re.sub(r"\{\{([^}]+)\}\}", repl, text)
    text = re.sub(r"\*\[([^\]]+)\]\*", r"[\1]", text)
    return text


def _resolve_image(path_str: str) -> Path | None:
    path_str = path_str.strip().split("{")[0].strip()
    candidates = [
        ROOT / path_str,
        ROOT / "reports" / "figures" / Path(path_str).name,
        FIG_DIR / Path(path_str).name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


class MdToPDF:
    def __init__(self, metrics: dict):
        from fpdf import FPDF

        self.metrics = metrics
        self.pdf = FPDF()
        self.pdf.set_margins(18, 18, 18)
        self.pdf.set_auto_page_break(auto=True, margin=18)
        self._unicode = False
        self._setup_font()

    def _setup_font(self):
        win = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        for regular, bold in [
            (win / "arial.ttf", win / "arialbd.ttf"),
            (win / "Arial.ttf", win / "Arialbd.ttf"),
        ]:
            if regular.exists():
                try:
                    self.pdf.add_font("F", "", str(regular))
                    self.pdf.add_font("F", "B", str(bold if bold.exists() else regular))
                    self._unicode = True
                    return
                except Exception:
                    pass

    def _set_font(self, style="", size=11):
        if self._unicode:
            self.pdf.set_font("F", style, size)
        else:
            self.pdf.set_font("Helvetica", style, size)

    def _w(self) -> float:
        return self.pdf.w - self.pdf.l_margin - self.pdf.r_margin

    def _write(self, text: str, h=5.5):
        text = text.replace("\r", "").strip()
        if not text:
            return
        self._set_font("", 11)
        self.pdf.set_x(self.pdf.l_margin)
        try:
            self.pdf.multi_cell(self._w(), h, text)
        except Exception:
            self.pdf.multi_cell(
                self._w(), h, text.encode("latin-1", "replace").decode("latin-1")
            )
        self.pdf.ln(1)

    def _heading(self, text: str, level: int):
        self.pdf.ln(3)
        size = {1: 15, 2: 13, 3: 11}.get(level, 11)
        self._set_font("B", size)
        self.pdf.set_x(self.pdf.l_margin)
        self.pdf.multi_cell(self._w(), 7, text)
        self.pdf.ln(2)

    def _add_image(self, path: Path):
        if not path.exists():
            self._write(f"[Figure manquante : {path.name}]")
            return
        try:
            w = self.pdf.w - self.pdf.l_margin - self.pdf.r_margin
            if self.pdf.get_y() > 200:
                self.pdf.add_page()
            self.pdf.image(str(path), w=min(w, 175))
            self.pdf.ln(4)
        except Exception as e:
            self._write(f"[Figure non integree : {path.name} ({e})]")

    def render_markdown(self, md_text: str):
        md_text = _fill_placeholders(md_text, self.metrics)
        lines = md_text.split("\n")
        in_code = False
        skip_until = 0

        for i, raw in enumerate(lines):
            if i < skip_until:
                continue
            line = raw.rstrip()

            if line.startswith("```"):
                in_code = not in_code
                if in_code:
                    self._write("(voir repository GitHub pour le code)")
                continue
            if in_code:
                continue

            if line.strip() == "---" or line.startswith("---"):
                self.pdf.ln(2)
                continue
            if line.startswith("\\newpage"):
                self.pdf.add_page()
                continue
            if line.startswith("!["):
                m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
                if m:
                    p = _resolve_image(m.group(2))
                    if p:
                        self._add_image(p)
                    else:
                        self._write(f"Figure : {Path(m.group(2)).name}")
                continue
            if line.startswith("# "):
                if self.pdf.page_no() == 1 and "Page de garde" in line:
                    continue
                self._heading(line[2:].strip(), 1)
                continue
            if line.startswith("## "):
                self._heading(line[3:].strip(), 2)
                continue
            if line.startswith("### "):
                self._heading(line[4:].strip(), 3)
                continue
            if line.startswith("|") and "|" in line[1:]:
                row = " | ".join(c.strip() for c in line.split("|")[1:-1])
                if row.replace("-", "").replace(":", "").strip():
                    self._write(row, h=5)
                continue
            if line.startswith("> "):
                self._write(line[2:].strip())
                continue
            if line.startswith("- ") or line.startswith("* "):
                self._write("- " + line[2:].strip())
                continue
            if re.match(r"^\d+\.\s", line):
                self._write(line.strip())
                continue
            if not line.strip():
                self.pdf.ln(2)
                continue
            clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            clean = re.sub(r"\*([^*]+)\*", r"\1", clean)
            clean = re.sub(r"`([^`]+)`", r"\1", clean)
            if clean.strip():
                self._write(clean)

    def cover_page(self):
        self.pdf.add_page()
        w = self._w()
        self._set_font("B", 17)
        self.pdf.ln(35)
        self.pdf.set_x(self.pdf.l_margin)
        self.pdf.multi_cell(w, 9, "Classification Robuste et Analyse de")
        self.pdf.set_x(self.pdf.l_margin)
        self.pdf.multi_cell(w, 9, "Decision en Environnement Critique")
        self.pdf.ln(10)
        self._set_font("", 13)
        self.pdf.set_x(self.pdf.l_margin)
        self.pdf.multi_cell(w, 8, "Prediction de pannes APS - Scania Trucks")
        self.pdf.ln(12)
        self._set_font("", 11)
        info = [
            "Methodologie : EDA, Elastic Net, Random Forest, XGBoost,",
            "Calibration, SHAP",
            "",
            f"Dataset : 60 000 train / 16 000 test | {self.metrics['n_pos']} pos / {self.metrics['n_neg']} neg",
            f"Desequilibre ~ 59:1 ({self.metrics['pos_pct']} % positifs)",
            f"Cout : FP=10 EUR | FN=500 EUR",
            "",
            f"Date : {date.today().strftime('%d/%m/%Y')}",
        ]
        for ln in info:
            self.pdf.set_x(self.pdf.l_margin)
            self.pdf.multi_cell(w, 7, ln)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.pdf.output(str(path))


def main():
    _ensure_fpdf()
    metrics = _load_metrics()
    print("Metriques chargees.")

    if not MD_PATH.exists():
        raise FileNotFoundError(MD_PATH)

    md = MD_PATH.read_text(encoding="utf-8")
    if md.startswith("---"):
        end = md.find("\n---", 4)
        if end > 0:
            md = md[end + 4 :]

    gen = MdToPDF(metrics)
    gen.cover_page()
    gen.render_markdown(md)
    gen.save(PDF_PATH)
    size_kb = PDF_PATH.stat().st_size / 1024
    print(f"PDF genere : {PDF_PATH} ({size_kb:.1f} Ko)")


if __name__ == "__main__":
    main()
