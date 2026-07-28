from __future__ import annotations
from typing import Sequence
import pandas as pd

from analysis.stats import Stats


class Analyzer:
    def __init__(self, alpha: float = 0.05, correction_method: str = "holm") -> None:
        self.alpha = alpha
        self.correction_method = correction_method
        self.stats = Stats(alpha)
        
    @staticmethod
    def validate_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
        missing = set(columns) - set(df.columns)
        if missing: raise ValueError(f"Missing columns: {sorted(missing)}") 
   
    def analyze_expe1(self, df: pd.DataFrame) -> dict:
        self.validate_columns(df, ["metric_type", "metric_name", "delta_perm_inter"])
        df = df.copy()
        df["delta_perm_inter"] = pd.to_numeric(df["delta_perm_inter"], errors="coerce")

        results: dict[str, dict[str, dict]] = {}

        grouped = df.groupby(["metric_type", "metric_name"], observed=True, sort=False)
        
        for (metric_type, metric_name), metric_df in grouped:

            result = self.stats.run_ttest(
                x=metric_df["delta_perm_inter"],
                y=0.0,
                test_name="perm_vs_inter",
                alternative="greater"
            )

            result.update({
                "metric_type": str(metric_type),
                "metric_name": str(metric_name),
                "mean_score_inter": float(metric_df["score_inter"].mean()),
                "mean_score_perm": float(metric_df["score_perm_mean"].mean()),
                "mean_delta_perm_inter": float(metric_df["delta_perm_inter"].mean()),
            })

            key = f"{metric_type}_{metric_name}"
            results[key] = result

        self.stats.apply_correction(
            test_results=list(results.values()),
            correction_method=self.correction_method,
        )

        return results
   
    def analyze_expe2(self, df: pd.DataFrame) -> dict:
        self.validate_columns(df, ["metric_type", "metric_name", "score_rgb", "score_ir", "delta_rgb_ir"])
        df = df.copy()

        numeric_columns = ["score_rgb", "score_ir", "delta_rgb_ir"]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

        results: dict[str, dict] = {}

        grouped = df.groupby(["metric_type", "metric_name"], observed=True, sort=False)
        
        for (metric_type, metric_name), metric_df in grouped:
            
            result = self.stats.run_ttest(
                x=metric_df["delta_rgb_ir"],
                y=0.0,
                test_name="rgb_vs_ir",
                alternative="two-sided",
            )

            result.update({
                "metric_type": str(metric_type),
                "metric_name": str(metric_name),
                "mean_score_rgb": float(metric_df["score_rgb"].mean()),
                "mean_score_ir": float(metric_df["score_ir"].mean()),
                "mean_delta_rgb_ir": float(metric_df["delta_rgb_ir"].mean()),
            })

            key = f"{metric_type}_{metric_name}"
            results[key] = result

        self.stats.apply_correction(
            test_results=list(results.values()),
            correction_method=self.correction_method,
        )

        return results
    
   

    def analyze_expe1_old(self, df: pd.DataFrame) -> dict:
        
        self.validate_columns(df, ["metric_type", "metric_name", "vv", "ii", "intra", "inter", "delta", "daynight"])
        df = df.copy()

        numeric_columns = ["vv", "ii", "intra", "inter", "delta"]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
        
        results: dict[str, dict[str, dict]] = {
            "consistency": {},
            "divergence": {},
            "daynight": {},
        }

        grouped = df.groupby(["metric_type", "metric_name"], observed=True, sort=False)
        
        for (metric_type, metric_name), metric_df in grouped:
            metric_type = str(metric_type)
            metric_name = str(metric_name)

            # Test 1 : ii against vv, paired by stem
            res_consistency = self.stats.run_ttest(x=metric_df["ii"], y=metric_df["vv"], test_name="ii_vs_vv", paired=True, alternative="two-sided")

            # Test 2 : intra against inter, paired by stem
            res_divergence = self.stats.run_ttest(x=metric_df["intra"], y=metric_df["inter"], test_name="intra_vs_inter", paired=True, alternative="greater")

            # Test 3 : comparison of the 2 daynight condition
            vis_a, vis_b = metric_df["daynight"].dropna().unique()
            val_a = metric_df.loc[metric_df["daynight"] == vis_a, "delta"]
            val_b = metric_df.loc[metric_df["daynight"] == vis_b, "delta"]
            res_daynight = self.stats.run_ttest(x=val_a, y=val_b, test_name=f"{vis_a}_vs_{vis_b}", paired=False, alternative="two-sided", correction=True)
            
            metric_info = {
                "metric_type": metric_type,
                "metric_name": metric_name,
            }
            res_consistency.update(metric_info)
            res_divergence.update(metric_info)
            res_daynight.update(metric_info)
            
            results["consistency"][metric_name] = res_consistency
            results["divergence"][metric_name] = res_divergence
            results["daynight"][metric_name] = res_daynight
            
        # Correction
        for test_type, metric_results in results.items():
            self.stats.apply_correction(
                test_results=list(metric_results.values()),
                correction_method=self.correction_method,
            )

        return results


    def analyze_expe2_old(self, df: pd.DataFrame) -> dict:
        
        self.validate_columns(df, ["metric_type", "metric_name", "delta_v", "delta_i", "delta", "daynight"])
        df = df.copy()

        numeric_columns = ["delta_v", "delta_i", "delta"]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

        configs = [
            ("visible", "delta_v", "greater"),
            ("infrared", "delta_i", "greater"),
            ("diff", "delta", "two-sided"),
        ]

        results: dict[str, dict[str, dict]] = {
            f"{prefix}_{test_type}": {}
            for prefix, _, _ in configs
            for test_type in ("effect", "daynight")
        }

        grouped = df.groupby(["metric_type", "metric_name"], observed=True, sort=False)

        for (metric_type, metric_name), metric_df in grouped:
            metric_type = str(metric_type)
            metric_name = str(metric_name)

            metric_results: dict = {}

            vis_a, vis_b = sorted(metric_df["daynight"].dropna().astype(str).unique())

            for prefix, column, alternative in configs:
                # Effet global : delta comparé à zéro
                effect_result = self.stats.run_ttest(
                    x=metric_df[column],
                    y=0,
                    test_name=f"{column}_vs_0",
                    paired=False,
                    alternative=alternative,
                )

                # Effet de la visibilité : comparaison des deux groupes
                val_a = metric_df.loc[metric_df["daynight"].astype(str) == vis_a,column]
                val_b = metric_df.loc[metric_df["daynight"].astype(str) == vis_b,column]
                daynight_result = self.stats.run_ttest(x=val_a, y=val_b, test_name=f"{prefix}_{vis_a}_vs_{vis_b}", paired=False, alternative="two-sided", correction=True)

                info = {
                    "metric_type": metric_type,
                    "metric_name": metric_name,
                    "variable": column,
                    "component": prefix,
                }
                effect_result.update(info)
                daynight_result.update(info)
                
                results[f"{prefix}_effect"][metric_name] = effect_result
                results[f"{prefix}_daynight"][metric_name] = daynight_result
                
        # Correction multiple séparée pour chaque famille de tests
        for test_type, metric_results in results.items():
            self.stats.apply_correction(
                test_results=list(metric_results.values()),
                correction_method=self.correction_method,
            )

        return results