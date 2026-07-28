from __future__ import annotations

import numpy as np
import pandas as pd
import pingouin as pg


class Stats:
    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def apply_correction(
        self,
        test_results: list[dict],
        correction_method: str,
    ) -> None:
        valid_results = [
            result
            for result in test_results
            if isinstance(result, dict)
            and result.get("p_value") is not None
            and np.isfinite(result["p_value"])
        ]
        if not valid_results:
            return

        reject, corrected_pvalues = pg.multicomp(
            [result["p_value"] for result in valid_results],
            alpha=self.alpha,
            method=correction_method,
        )

        for result, corrected_p, is_significant in zip(
            valid_results,
            corrected_pvalues,
            reject,
        ):
            result.update({
                "p_value_corrected": float(corrected_p),
                "significant_corrected": bool(is_significant),
                "correction_method": correction_method,
            })

    def run_ttest(
        self,
        x: pd.Series,
        y: pd.Series | float,
        test_name: str,
        paired: bool = False,
        alternative: str = "two-sided",
        correction: str | bool = "auto",
        r: float = 0.707,
        confidence: float = 0.95,
    ) -> dict:
        x = pd.to_numeric(x, errors="coerce")

        if np.isscalar(y):
            x = x.dropna()
            y = float(y)
            delta = x - y

            base = {
                "test": test_name,
                "test_type": "one-sample",
                "alternative": alternative,
                "n": int(len(x)),
                "mean_x": float(x.mean()) if len(x) else np.nan,
                "mean_y": y,
                "mean_diff": float(delta.mean()) if len(x) else np.nan,
                "std_x": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
                "std_y": 0.0,
                "std_diff": (
                    float(delta.std(ddof=1))
                    if len(delta) > 1
                    else np.nan
                ),
                "significant": False,
            }

            if len(x) < 2:
                return base

        elif paired:
            y = pd.to_numeric(y, errors="coerce")
            paired_data = pd.concat(
                [x.rename("x"), y.rename("y")],
                axis=1,
            ).dropna()

            x = paired_data["x"]
            y = paired_data["y"]
            delta = x - y

            base = {
                "test": test_name,
                "test_type": "paired",
                "alternative": alternative,
                "n": int(len(delta)),
                "mean_x": float(x.mean()) if len(delta) else np.nan,
                "mean_y": float(y.mean()) if len(delta) else np.nan,
                "mean_diff": float(delta.mean()) if len(delta) else np.nan,
                "std_x": float(x.std(ddof=1)) if len(delta) > 1 else np.nan,
                "std_y": float(y.std(ddof=1)) if len(delta) > 1 else np.nan,
                "std_diff": (
                    float(delta.std(ddof=1))
                    if len(delta) > 1
                    else np.nan
                ),
                "significant": False,
            }

            if len(delta) < 2:
                return base

        else:
            y = pd.to_numeric(y, errors="coerce")
            x = x.dropna()
            y = y.dropna()

            base = {
                "test": test_name,
                "test_type": "independent",
                "alternative": alternative,
                "n_x": int(len(x)),
                "n_y": int(len(y)),
                "mean_x": float(x.mean()) if len(x) else np.nan,
                "mean_y": float(y.mean()) if len(y) else np.nan,
                "mean_diff": (
                    float(x.mean() - y.mean())
                    if len(x) and len(y)
                    else np.nan
                ),
                "std_x": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
                "std_y": float(y.std(ddof=1)) if len(y) > 1 else np.nan,
                "significant": False,
            }

            if len(x) < 2 or len(y) < 2:
                return base

        res = pg.ttest(
            x=x,
            y=y,
            paired=paired,
            alternative=alternative,
            correction=correction,
            r=r,
            confidence=confidence,
        ).iloc[0]

        base.update({
            "T": float(res["T"]),
            "dof": float(res["dof"]),
            "p_value": float(res["p_val"]),
            "cohen_d": float(res["cohen_d"]),
            "significant": float(res["p_val"]) < self.alpha,
        })

        if "CI95" in res.index:
            ci = res["CI95"]
            base["CI95"] = ci.tolist() if hasattr(ci, "tolist") else ci

        if "power" in res.index and pd.notna(res["power"]):
            base["power"] = float(res["power"])

        if "BF10" in res.index and pd.notna(res["BF10"]):
            try:
                base["BF10"] = float(res["BF10"])
            except (TypeError, ValueError):
                base["BF10"] = str(res["BF10"])

        return base

    def run_anova(
        self,
        df: pd.DataFrame,
        value_col: str,
        factor_col: str,
    ) -> dict:
        sub = df[[value_col, factor_col]].dropna()
        n_groups = sub[factor_col].nunique()
        base = {
            "test": f"{value_col} ~ {factor_col}",
            "n": int(len(sub)),
            "n_groups": int(n_groups),
            "significant": False,
        }

        if len(sub) < 3 or n_groups < 2:
            return base

        res = pg.anova(
            data=sub,
            dv=value_col,
            between=factor_col,
            detailed=True,
            effsize="np2",
        )
        row = res.loc[res["Source"] == factor_col].iloc[0]
        error_rows = res.loc[res["Source"].isin(["Within", "Residual"])]

        base.update({
            "F": float(row["F"]),
            "ddof1": float(row["DF"]),
            "p_value": float(row["p_unc"]),
            "significant": float(row["p_unc"]) < self.alpha,
        })

        if not error_rows.empty:
            base["ddof2"] = float(error_rows["DF"].iloc[0])
        if "np2" in row.index:
            base["np2"] = float(row["np2"])

        return base
