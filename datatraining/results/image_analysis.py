from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
    from skimage.feature import hog, canny
    from skimage import filters, exposure, color
    HAVE_SKI = True
except Exception:
    HAVE_SKI = False

try:
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

def extract_features_for_visual(img, use_hog=True, preprocess=True, threshold_segmentation=True):
    intermediates = {}
    img = img.convert("RGB")
    arr_rgb = np.array(img); intermediates["rgb"] = arr_rgb
    if preprocess and HAVE_SKI:
        gray = color.rgb2gray(arr_rgb); intermediates["grayscale"] = gray
        gray_smooth = filters.gaussian(gray, sigma=1.0); intermediates["gaussian_smoothing"] = gray_smooth
        edges = canny(gray_smooth, sigma=1.0); intermediates["canny_edges"] = edges.astype(np.uint8)
        if threshold_segmentation:
            thresh = filters.threshold_otsu(gray_smooth)
            mask = (gray_smooth > thresh).astype(np.uint8); gray_proc = gray_smooth * mask
        else:
            mask = np.ones_like(gray_smooth, dtype=np.uint8); gray_proc = gray_smooth
        intermediates["otsu_threshold_mask"] = mask; intermediates["thresholded_region"] = gray_proc
        arr_proc = np.stack([gray_proc, edges.astype(float), gray_smooth], axis=-1)
        arr_proc_u8 = exposure.rescale_intensity(arr_proc, out_range=(0, 255)).astype(np.uint8)
        intermediates["combined_feature_map"] = arr_proc_u8
    else:
        arr_proc_u8 = arr_rgb
        for k in ("grayscale","gaussian_smoothing","canny_edges","otsu_threshold_mask","thresholded_region"):
            intermediates[k] = None
        intermediates["combined_feature_map"] = arr_proc_u8

    bins = 32; feats_parts=[]; hist_data=[]
    for c in range(arr_proc_u8.shape[2]):
        h, edges_bins = np.histogram(arr_proc_u8[:,:,c], bins=bins, range=(0,255), density=True)
        feats_parts.append(h); hist_data.append((h, edges_bins))
    feats = np.concatenate(feats_parts).astype("float32"); intermediates["color_hist_data"]=hist_data

    hog_vec = None; hog_image = None
    if use_hog and HAVE_SKI:
        try:
            gray_for_hog = color.rgb2gray(arr_proc_u8)
            hog_vec, hog_img = hog(gray_for_hog, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True, visualize=True)
            feats = np.concatenate([feats, hog_vec.astype("float32")])
            hog_image = exposure.rescale_intensity(hog_img, out_range=(0,255)).astype(np.uint8)
        except Exception as e:
            warnings.warn(f"HOG failed: {e}")
    intermediates["hog_vector"]=hog_vec; intermediates["hog_image"]=hog_image
    return feats, intermediates

def _ensure_out(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True); return out_dir

def _save_fig(fig_path: Path, make_plot):
    plt.figure(); make_plot(); plt.tight_layout(); plt.savefig(fig_path, dpi=200); plt.close()

def save_image(arr, out_path: Path, cmap=None):
    def plot(): 
        if arr.ndim==2: plt.imshow(arr, cmap=cmap or "gray")
        else: plt.imshow(arr); plt.axis("off")
    _save_fig(out_path, plot)

def plot_histograms(hist_data, out_path: Path, title="Color/Channel Histograms"):
    def plot():
        for i,(h,edges) in enumerate(hist_data):
            centers = 0.5*(edges[:-1]+edges[1:]); plt.plot(centers,h,label=f"channel {i+1}")
        plt.title(title); plt.xlabel("Intensity"); plt.ylabel("Density"); plt.legend()
    _save_fig(out_path, plot)

def plot_bar(values, out_path: Path, title: str, max_bars: int = 256):
    vals = np.asarray(values).ravel()
    if len(vals) > max_bars:
        vals = vals[:max_bars]; subtitle = f"(first {max_bars} of {len(values)} dims)"
    else:
        subtitle = f"({len(values)} dims)"
    def plot(): 
        plt.figure(figsize=(10,3)); plt.bar(np.arange(len(vals)), vals, width=1.0); plt.title(f"{title} {subtitle}"); plt.xlabel("Feature index"); plt.ylabel("Value")
    _save_fig(out_path, plot)

def plot_probs(classes, probs, out_path: Path, title="Class probabilities"):
    idx = np.argsort(-probs); classes_sorted=[classes[i] for i in idx]; probs_sorted=probs[idx]
    def plot(): 
        plt.barh(range(len(classes_sorted)), probs_sorted); plt.yticks(range(len(classes_sorted)), classes_sorted); plt.gca().invert_yaxis(); plt.xlabel("Probability"); plt.title(title)
    _save_fig(out_path, plot)

def save_prediction_overlay(img_rgb: np.ndarray, label: str, conf: float, out_path: Path):
    def plot():
        plt.imshow(img_rgb); plt.axis("off")
        txt = f"Prediction: {label}\nConfidence: {conf:.3f}"
        plt.gcf().text(0.02, 0.02, txt, fontsize=12, bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.3"))
    _save_fig(out_path, plot)

def visualize(image_path: Path, model_path: Path | None, out_dir: Path):
    out_dir = _ensure_out(out_dir)
    img = Image.open(image_path).convert("RGB").resize((128,128))
    feats, inter = extract_features_for_visual(img, use_hog=True, preprocess=True, threshold_segmentation=True)
    if inter.get("rgb") is not None: save_image(inter["rgb"], out_dir/"image_rgb.png")
    if inter.get("grayscale") is not None: save_image(inter["grayscale"], out_dir/"grayscale.png", cmap="gray")
    if inter.get("gaussian_smoothing") is not None: save_image(inter["gaussian_smoothing"], out_dir/"gaussian_smoothing.png", cmap="gray")
    if inter.get("canny_edges") is not None: save_image(inter["canny_edges"]*255, out_dir/"canny_edges.png", cmap="gray")
    if inter.get("otsu_threshold_mask") is not None: save_image(inter["otsu_threshold_mask"]*255, out_dir/"otsu_threshold_mask.png", cmap="gray")
    if inter.get("thresholded_region") is not None: save_image(inter["thresholded_region"], out_dir/"thresholded_region.png", cmap="gray")
    if inter.get("combined_feature_map") is not None: save_image(inter["combined_feature_map"], out_dir/"combined_feature_map.png")
    if inter.get("color_hist_data"): plot_histograms(inter["color_hist_data"], out_dir/"color_histograms.png", title="Channel Histograms (proc | edges | smooth)")
    if inter.get("hog_image") is not None: save_image(inter["hog_image"], out_dir/"hog_gradients.png", cmap="gray")
    plot_bar(feats, out_dir/"feature_vector.png", title="Feature vector (histograms + HOG)")

    meta={}
    if model_path is not None and HAVE_SKLEARN and Path(model_path).exists():
        payload = joblib.load(model_path); model = payload.get("model", None); classes = payload.get("classes", None)
        scaler = None
        if isinstance(model, Pipeline):
            scaler = model.named_steps.get("standardscaler", None)
            if scaler is None:
                for _,step in model.named_steps.items():
                    if isinstance(step, StandardScaler):
                        scaler = step; break
        if scaler is not None:
            z = scaler.transform([feats])[0]; plot_bar(z, out_dir/"standardized_features.png", title="Standardized features (z-scores)"); meta["standardized_dims"]=int(z.shape[0])
        else:
            meta["standardized_dims"]=None

        probs=None; pred_label=None; conf=None
        try:
            probs = model.predict_proba([feats])[0]; pred_idx=int(np.argmax(probs))
            pred_label = classes[pred_idx] if classes is not None else str(pred_idx); conf=float(probs[pred_idx])
        except Exception:
            try:
                pred = model.predict([feats])[0]; pred_label=str(pred); conf=0.5
            except Exception:
                pred_label="unknown"; conf=0.0

        if probs is not None and classes is not None: plot_probs(classes, probs, out_dir/"class_probabilities.png", title="Random Forest: P(class|x)")
        try:
            big = Image.open(image_path).convert("RGB"); save_prediction_overlay(np.array(big), pred_label, conf, out_dir/"prediction_overlay.png")
        except Exception:
            save_prediction_overlay(inter["rgb"], pred_label, conf, out_dir/"prediction_overlay.png")
        meta["prediction"] = {"label": pred_label, "confidence": conf}

    meta.update({
        "image_path": str(image_path),
        "out_dir": str(out_dir),
        "have_skimage": HAVE_SKI,
        "have_sklearn": HAVE_SKLEARN,
        "feature_dims": int(feats.shape[0]),
        "artifacts": [
            "image_rgb.png","grayscale.png","gaussian_smoothing.png","canny_edges.png","otsu_threshold_mask.png",
            "thresholded_region.png","combined_feature_map.png","color_histograms.png","hog_gradients.png","feature_vector.png",
            "standardized_features.png","class_probabilities.png","prediction_overlay.png",
        ],
    })

    with open(out_dir/"summary_single_image.json","w") as f: json.dump(meta,f,indent=2)
    print("✅ Saved visualizations to:", out_dir)

def parse_args():
    p = argparse.ArgumentParser(description="Visualize SmartTrashSorter math/vision stages for a single image")
    p.add_argument("--image", required=True, type=str)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    img_path = Path(args.image)
    out_dir = Path(args.out) if args.out else (img_path.parent/"analysis_single")
    model_path = Path(args.model) if args.model else None
    visualize(img_path, model_path, out_dir)
