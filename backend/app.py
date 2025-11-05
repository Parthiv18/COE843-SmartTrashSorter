from __future__ import annotations
import io, joblib, time
from pathlib import Path
from typing import Any, Dict, List, Optional
from flask import Flask, jsonify, render_template, request
from PIL import Image
from datatraining.results.model import TrashModel

BASEDIR = Path(__file__).resolve().parent

def _load_payload(model_path: str) -> Optional[Dict[str, Any]]:
    try:
        payload = joblib.load(model_path)
        if isinstance(payload, dict): return payload
    except Exception:
        pass
    return None

def _build_dataset_metrics_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not payload: return {}
    def _safe_convert(x):
        try:
            import numpy as _np
        except Exception:
            _np = None
        try:
            if _np is not None and isinstance(x, _np.ndarray):
                if x.size > 1000: return {"shape": tuple(x.shape)}
                return x.tolist()
        except Exception:
            pass
        try:
            import pandas as _pd
            if isinstance(x, _pd.DataFrame): return {"shape": x.shape, "columns": list(x.columns)}
            if isinstance(x, _pd.Series): return {"shape": x.shape, "name": x.name}
        except Exception:
            pass
        return x

    metrics: Dict[str, Any] = {}
    for key in ("accuracy", "per_class_metrics", "confusion_matrix", "confusion_matrix_norm", "created_at", "classes"):
        if key in payload: metrics[key] = _safe_convert(payload[key])
    if "filenames_test" in payload:
        ft = payload.get("filenames_test")
        try: metrics["filenames_test_sample"] = list(ft)[:20]
        except Exception: metrics["filenames_test_sample"] = ft
    if "X_test" in payload:
        xt = payload.get("X_test")
        try: metrics["X_test_shape"] = getattr(xt, "shape", None)
        except Exception: metrics["X_test_shape"] = None
    return metrics

def _build_sample_result_from_payload(model_obj: TrashModel, predictions: List[Dict[str, Any]], true_label: Optional[str], payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"predictions": predictions, "true_label": true_label}
    try:
        classes = getattr(model_obj, "classes", None)
        if classes is not None:
            try: result["model_info"] = {"loaded_classes": list(classes)}
            except Exception: result["model_info"] = {"loaded_classes": classes}
    except Exception:
        pass
    if payload:
        def sc(key):
            val = payload.get(key)
            try:
                import numpy as _np
                if isinstance(val, _np.ndarray):
                    if val.size > 1000: return {"shape": tuple(val.shape)}
                    return val.tolist()
            except Exception:
                pass
            return val
        sample_summary: Dict[str, Any] = {}
        sample_summary["accuracy"] = sc("accuracy")
        sample_summary["created_at"] = payload.get("created_at")
        sample_summary["per_class_metrics"] = payload.get("per_class_metrics")
        sample_summary["confusion_matrix"] = sc("confusion_matrix")
        sample_summary["confusion_matrix_norm"] = sc("confusion_matrix_norm")
        xt = payload.get("X_test")
        try: sample_summary["X_test_shape"] = getattr(xt, "shape", None)
        except Exception: sample_summary["X_test_shape"] = None
        result["dataset_summary"] = sample_summary
    return result

def create_app(test_config: Optional[Dict[str, Any]] = None) -> Flask:
    template_folder = str(BASEDIR.parent / "frontend" / "templates")
    static_folder = str(BASEDIR.parent / "frontend" / "static")
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
    app.config.from_mapping(MODEL_PATH=str(BASEDIR.parent / "datatraining" / "results" / "trained_model.joblib"))
    if test_config: app.config.update(test_config)
    model_path = app.config.get("MODEL_PATH")
    model = TrashModel(model_path)
    payload = _load_payload(model_path)
    dataset_metrics = _build_dataset_metrics_from_payload(payload) if payload else None

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/predict", methods=["POST"])
    def predict():
            if "image" not in request.files: return jsonify({"error": "missing image file"}), 400
            file = request.files["image"]
            if file.filename == "": return jsonify({"error": "empty filename"}), 400
            true_label = request.form.get("true_label") or request.args.get("true_label") or None
            try:
                img_bytes = file.read()
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as exc:
                return jsonify({"error": "invalid image", "details": str(exc)}), 400
            try:
                pred = model.predict(pil_img)
            except Exception as exc:
                return jsonify({"error": "prediction failed", "details": str(exc)}), 500

            predictions: List[Dict[str, Any]] = []
            if isinstance(pred, list):
                for item in pred:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        lbl, conf = item[0], item[1]
                        try: conf = float(conf)
                        except Exception: conf = 0.0
                        predictions.append({"label": str(lbl), "confidence": conf})
                    else:
                        predictions.append({"label": str(item), "confidence": 0.0})
            elif isinstance(pred, tuple) and len(pred) >= 2:
                label, conf = pred[0], pred[1]
                try: conf = float(conf)
                except Exception: conf = 0.0
                predictions.append({"label": str(label), "confidence": conf})
            else:
                predictions.append({"label": str(pred), "confidence": 0.0})

            try:
                sample_result = _build_sample_result_from_payload(model, predictions, true_label, payload)
            except Exception:
                sample_result = {"predictions": predictions, "true_label": true_label}

            if dataset_metrics: sample_result["dataset_metrics"] = dataset_metrics
            return jsonify(sample_result)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
