import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


class SingleClassModel:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.column_stack([
            np.ones(len(X), dtype=float),
            np.zeros(len(X), dtype=float),
        ])


class ThresholdModel:
    def predict_proba(self, X):
        return np.array([[0.4, 0.6]], dtype=float)


class PipelineFixTests(unittest.TestCase):

    def test_evaluate_model_handles_single_class_y_test(self):
        try:
            from src.evaluation.metrics import evaluate_model
        except ImportError as exc:
            self.skipTest(f"metrics dependencies are not installed: {exc}")

        X = pd.DataFrame({"feature": [0.1, 0.2, 0.3]})
        y = pd.Series([0, 0, 0])

        results = evaluate_model(SingleClassModel(), X, y)

        self.assertIsNone(results["roc_auc"])
        self.assertEqual(results["confusion_matrix"], [[3, 0], [0, 0]])
        self.assertEqual(results["f1_score"], 0.0)

    def test_sample_training_data_keeps_minority_classes(self):
        from src.training.sample_training_data import sample_training_data

        df = pd.DataFrame({
            "feature": range(101),
            "Class": [0] * 100 + [1],
        })

        sampled = sample_training_data(df, "Class", max_samples=10, random_state=42)

        self.assertEqual(len(sampled), 10)
        self.assertEqual(set(sampled["Class"]), {0, 1})

    def test_select_best_model_info_raises_clear_error_for_empty_registry(self):
        from src.registry.best_model_selector import select_best_model_info

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "No valid model metadata"):
                select_best_model_info(tmp)

    def test_select_best_model_info_requires_matching_pickle(self):
        from src.registry.best_model_selector import select_best_model_info

        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp)
            metadata = {
                "metrics": {"roc_auc": 0.9},
                "features": ["B02"],
            }
            (registry / "model_20260101_000000.json").write_text(json.dumps(metadata))

            with self.assertRaisesRegex(FileNotFoundError, "missing file"):
                select_best_model_info(registry)

    def test_aggregate_clears_doy_where_final_mask_is_zero(self):
        try:
            import rasterio
            from rasterio.transform import from_origin
            from src.postprocessing.aggregate_tile_outputs import aggregate_tile
        except ImportError:
            self.skipTest("rasterio is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tile = "T44QME"
            processed = root / "processed" / tile
            final = root / "final"
            processed.mkdir(parents=True)

            meta = {
                "driver": "GTiff",
                "height": 2,
                "width": 2,
                "count": 1,
                "dtype": "float32",
                "transform": from_origin(0, 2, 1, 1),
                "crs": "EPSG:4326",
            }

            dates = ["20260101", "20260110"]
            masks = [
                np.array([[1, 0], [0, 0]], dtype=np.uint8),
                np.array([[1, 0], [0, 0]], dtype=np.uint8),
            ]

            for date, mask in zip(dates, masks):
                with rasterio.open(processed / f"{tile}_{date}_BurntProb.tif", "w", **meta) as dst:
                    dst.write(mask.astype("float32"), 1)
                mask_meta = {**meta, "dtype": "uint8"}
                with rasterio.open(processed / f"{tile}_{date}_BurntMask.tif", "w", **mask_meta) as dst:
                    dst.write(mask, 1)

            def fake_mmu(mask, min_pixels):
                return np.zeros_like(mask)

            config = {
                "postprocessing": {
                    "temporal_consistency": {"enabled": True, "window": 1},
                    "mmu": {"enabled": True, "pixels": 99},
                    "masks": {
                        "mask_path": str(root / "aux"),
                        "use_water": False,
                        "use_urban": False,
                        "use_forest": False,
                        "use_agriculture": False,
                    },
                }
            }

            with patch("src.postprocessing.aggregate_tile_outputs.apply_mmu", fake_mmu):
                aggregate_tile(tile, root / "processed", final, config)

            with rasterio.open(final / tile / f"{tile}_FirstBurnDOY.tif") as src:
                doy = src.read(1)

            self.assertTrue(np.all(doy == 0))

    def test_predict_tile_can_use_configured_threshold(self):
        try:
            import src.inference.predict_tile as predict_module
        except ImportError as exc:
            self.skipTest(f"inference dependencies are not installed: {exc}")

        feature_names = [
            "B11", "B12", "B02", "B03", "B04", "B08", "BAI", "GEMI",
            "MIRBI", "NBR1", "NBR2", "NDMI", "NDVI", "NDWI", "SAVI",
        ]
        feature_stack = np.ones((len(feature_names), 1, 1), dtype=float)
        written = {}

        class FakeLoader:
            def __init__(self, safe_path):
                pass

            def load_bands(self, bands):
                stack = np.ones((6, 1, 1), dtype=float)
                meta = {
                    "driver": "GTiff",
                    "height": 1,
                    "width": 1,
                    "count": 1,
                    "dtype": "float32",
                }
                return stack, meta, np.zeros((1, 1), dtype=bool)

        class FakeRaster:
            def __init__(self, path, mode, **meta):
                self.path = Path(path).name

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def write(self, data, *args):
                written[self.path] = np.array(data)

        with tempfile.TemporaryDirectory() as tmp:
            safe = Path(tmp) / "S2A_MSIL2A_20260101T000000_N0000_R000_T44QME_20260101T000000.SAFE"
            safe.mkdir()

            with patch.object(predict_module, "select_best_model_info", return_value=("model.pkl", {"metrics": {"optimal_threshold": 0.7}, "features": feature_names})), \
                 patch.object(predict_module.joblib, "load", return_value=ThresholdModel()), \
                 patch.object(predict_module, "Sentinel2L2ALoader", FakeLoader), \
                 patch.object(predict_module, "build_feature_stack", return_value=(feature_stack, feature_names)), \
                 patch.object(predict_module.rasterio, "open", FakeRaster):

                predict_module.predict_tile(
                    str(safe),
                    "registry",
                    threshold=0.5,
                    output_dir=tmp,
                    use_registered_threshold=False,
                )

        mask = written["T44QME_20260101_BurntMask.tif"]
        self.assertEqual(int(mask[0, 0]), 1)

    def test_predict_tile_can_use_registered_threshold(self):
        try:
            import src.inference.predict_tile as predict_module
        except ImportError as exc:
            self.skipTest(f"inference dependencies are not installed: {exc}")

        feature_names = [
            "B11", "B12", "B02", "B03", "B04", "B08", "BAI", "GEMI",
            "MIRBI", "NBR1", "NBR2", "NDMI", "NDVI", "NDWI", "SAVI",
        ]
        feature_stack = np.ones((len(feature_names), 1, 1), dtype=float)
        written = {}

        class FakeLoader:
            def __init__(self, safe_path):
                pass

            def load_bands(self, bands):
                stack = np.ones((6, 1, 1), dtype=float)
                meta = {
                    "driver": "GTiff",
                    "height": 1,
                    "width": 1,
                    "count": 1,
                    "dtype": "float32",
                }
                return stack, meta, np.zeros((1, 1), dtype=bool)

        class FakeRaster:
            def __init__(self, path, mode, **meta):
                self.path = Path(path).name

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def write(self, data, *args):
                written[self.path] = np.array(data)

        with tempfile.TemporaryDirectory() as tmp:
            safe = Path(tmp) / "S2A_MSIL2A_20260101T000000_N0000_R000_T44QME_20260101T000000.SAFE"
            safe.mkdir()

            with patch.object(predict_module, "select_best_model_info", return_value=("model.pkl", {"metrics": {"optimal_threshold": 0.7}, "features": feature_names})), \
                 patch.object(predict_module.joblib, "load", return_value=ThresholdModel()), \
                 patch.object(predict_module, "Sentinel2L2ALoader", FakeLoader), \
                 patch.object(predict_module, "build_feature_stack", return_value=(feature_stack, feature_names)), \
                 patch.object(predict_module.rasterio, "open", FakeRaster):

                predict_module.predict_tile(
                    str(safe),
                    "registry",
                    threshold=0.5,
                    output_dir=tmp,
                    use_registered_threshold=True,
                )

        mask = written["T44QME_20260101_BurntMask.tif"]
        self.assertEqual(int(mask[0, 0]), 0)


if __name__ == "__main__":
    unittest.main()
