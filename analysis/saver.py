from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from helpers.models import Results


class Saver:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        
    

    def save_expe1_results(
        self,
        results: Results,
        output_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        output_dir = (
            self.output_dir / "expe1" / "comparison"
            if output_dir is None
            else Path(output_dir)
        )

        expected_columns = [
            "stem",
            "metric_type",
            "metric_name",
            "score_inter",
            "score_perm_mean",
            "score_perm_std",
            "delta_perm_inter",
            "p_value",
            "n_permutations",
            "n_visible",
            "n_infrared",
            "sigma",
        ]

        return self._save_results(
            results=results,
            output_dir=output_dir,
            expected_columns=expected_columns,
            summary_measures=[
                "score_inter",
                "score_perm_mean",
                "score_perm_std",
                "delta_perm_inter",
            ],
        )


    def save_expe2_results(
        self,
        results: Results,
        output_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        output_dir = (
            self.output_dir / "expe2" / "comparison"
            if output_dir is None
            else Path(output_dir)
        )

        expected_columns = [
            "stem",
            "metric_type",
            "metric_name",
            "score_rgb",
            "score_ir",
            "delta_rgb_ir",
        ]

        return self._save_results(
            results=results,
            output_dir=output_dir,
            expected_columns=expected_columns,
            summary_measures=[
                "score_rgb",
                "score_ir",
                "delta_rgb_ir",
            ],
        )


    def save_expe1_analysis(
        self,
        analysis: Mapping[str, Any],
        output_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        output_dir = (
            self.output_dir / "expe1" / "statistics"
            if output_dir is None
            else Path(output_dir)
        )

        return self._save_analysis(
            analysis=analysis,
            output_dir=output_dir,
        )


    def save_expe2_analysis(
        self,
        analysis: Mapping[str, Any],
        output_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        output_dir = (
            self.output_dir / "expe2" / "statistics"
            if output_dir is None
            else Path(output_dir)
        )

        return self._save_analysis(
            analysis=analysis,
            output_dir=output_dir,
        )

    def _save_analysis_json(
        self,
        analysis: Mapping[str, Any],
        output_path: Path,
    ) -> None:
        if not isinstance(analysis, Mapping):
            raise TypeError(
                "analysis must be a mapping, "
                f"got {type(analysis).__name__}."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                self._to_json_compatible(analysis),
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )


    def _to_csv_value(self, value: Any) -> Any:
        """
        Convertit une valeur de résultat en cellule compatible avec un CSV.

        Les objets comme les intervalles de confiance sont stockés sous
        forme de JSON dans une seule cellule.
        """
        if self._is_scalar_value(value):
            return self._to_python_scalar(value)

        return json.dumps(
            self._to_json_compatible(value),
            ensure_ascii=False,
        )


    @staticmethod
    def _is_scalar_value(value: Any) -> bool:
        return value is None or isinstance(
            value,
            (str, bytes, bool, int, float, np.generic, pd.Timestamp),
        )

    @classmethod
    def _to_json_compatible(cls, value: Any) -> Any:
        if isinstance(value, pd.DataFrame):
            return [
                {str(key): cls._to_json_compatible(item) for key, item in row.items()}
                for row in value.to_dict(orient="records")
            ]
        if isinstance(value, pd.Series):
            return {
                str(key): cls._to_json_compatible(item)
                for key, item in value.to_dict().items()
            }
        if isinstance(value, Mapping):
            return {
                str(key): cls._to_json_compatible(item)
                for key, item in value.items()
            }
        if isinstance(value, np.ndarray):
            return cls._to_json_compatible(value.tolist())
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return [cls._to_json_compatible(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, np.generic):
            return cls._to_json_compatible(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        return str(value)

    @staticmethod
    def _to_python_scalar(value: Any) -> Any:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, float) and not np.isfinite(value):
            return np.nan
        return value

    def _save_results(
        self,
        results: Results,
        output_dir: Path,
        expected_columns: list[str],
        summary_measures: list[str],
    ) -> pd.DataFrame:

        output_dir.mkdir(parents=True, exist_ok=True)

        summary_rows: list[dict[str, Any]] = []
        all_dataframes: list[pd.DataFrame] = []

        for metric_type, metrics in results.items():
            metric_type_dir = output_dir / metric_type
            metric_type_dir.mkdir(parents=True, exist_ok=True)

            for metric_name, stem_results in metrics.items():
                rows: list[dict[str, Any]] = []

                for stem, values in stem_results.items():
                    if values is None:
                        continue

                    row = {
                        "stem": str(stem),
                        "metric_type": metric_type,
                        "metric_name": metric_name,
                    }

                    row.update(values)
                    rows.append(row)

                df = pd.DataFrame(rows)

                if df.empty:
                    df = pd.DataFrame(columns=expected_columns)
                else:
                    df = self._ensure_columns(df, expected_columns)
                    all_dataframes.append(df)

                # Un CSV par métrique
                metric_path = metric_type_dir / f"{metric_name}.csv"
                df.to_csv(metric_path, index=False)

                # Résumé moyen de la métrique
                summary_row = {
                    "metric_type": metric_type,
                    "metric_name": metric_name,
                }

                for measure in summary_measures:
                    summary_row[measure] = (
                        self._safe_mean(df[measure])
                        if measure in df.columns
                        else np.nan
                    )

                summary_rows.append(summary_row)

        # Résumé par métrique
        summary_columns = [
            "metric_type",
            "metric_name",
            *summary_measures,
        ]

        summary = pd.DataFrame(summary_rows)
        summary = self._ensure_columns(summary, summary_columns)
        summary.to_csv(output_dir / "summary.csv", index=False)

        # Toutes les observations individuelles
        if all_dataframes:
            all_results = pd.concat(
                all_dataframes,
                ignore_index=True,
            )
            all_results = self._ensure_columns(
                all_results,
                expected_columns,
            )
        else:
            all_results = pd.DataFrame(columns=expected_columns)

        all_results.to_csv(
            output_dir / "all_results.csv",
            index=False,
        )

        return all_results

    @staticmethod
    def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        for column in columns:
            if column not in df.columns:
                df[column] = np.nan

        return df[columns]

    @staticmethod
    def _safe_mean(series: pd.Series) -> float:
        values = pd.to_numeric(series, errors="coerce")
        if values.dropna().empty:
            return float("nan")
        return float(values.mean())
    
    
    def _save_analysis(
        self,
        analysis: Mapping[str, Any],
        output_dir: Path,
    ) -> pd.DataFrame:
        output_dir.mkdir(parents=True, exist_ok=True)

        self._save_analysis_json(
            analysis=analysis,
            output_path=output_dir / "analysis.json",
        )

        rows: list[dict[str, Any]] = []

        for metric_key, result in analysis.items():
            if not isinstance(result, Mapping):
                continue

            row = {
                "metric_key": str(metric_key),
                "metric_type": result.get("metric_type"),
                "metric_name": result.get("metric_name"),
                "test_name": result.get(
                    "test",
                    result.get("test_name"),
                ),
            }

            for key, value in result.items():
                if key in {
                    "metric_type",
                    "metric_name",
                    "test",
                    "test_name",
                }:
                    continue

                row[str(key)] = self._to_csv_value(value)

            rows.append(row)

        table = pd.DataFrame(rows)

        first_columns = [
            "metric_key",
            "metric_type",
            "metric_name",
            "test_name",
        ]

        if table.empty:
            table = pd.DataFrame(columns=first_columns)
        else:
            remaining_columns = [
                column
                for column in table.columns
                if column not in first_columns
            ]

            table = table[first_columns + remaining_columns]

        table.to_csv(
            output_dir / "analysis.csv",
            index=False,
        )

        return table