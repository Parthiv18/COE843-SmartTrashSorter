from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, label_binarize
    from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, average_precision_score
    from sklearn.manifold import TSNE
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

def _ensure_out(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True); return out_dir

def _savefig(path: Path):
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

def _barh(vals, labels, title, out_path: Path):
    idx = np.argsort(vals); v = np.array(vals)[idx]; l = np.array(labels)[idx]
    plt.figure(figsize=(7, max(3, 0.28*len(v)))); plt.barh(range(len(v)), v); plt.yticks(range(len(v)), l); plt.title(title); plt.xlabel("Value"); _savefig(out_path)

def _table_image(headers, rows, title, out_path: Path):
    fig, ax = plt.subplots(figsize=(max(6, 0.5*len(headers)+2), max(3, 0.35*len(rows)+2))); ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=headers, loc='center'); tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1,1.3)
    ax.set_title(title, pad=12); _savefig(out_path)

def _confusion_pairs(cm: np.ndarray, classes):
    pairs = [(classes[i], classes[j], int(cm[i,j])) for i in range(cm.shape[0]) for j in range(cm.shape[1]) if i!=j and cm[i,j]>0]
    pairs.sort(key=lambda x: -x[2]); return pairs

def _safe_get_pipeline_step(model, cls_type):
    if isinstance(model, Pipeline):
        for _, step in model.named_steps.items():
            if isinstance(step, cls_type): return step
    return None

def analyze(model_path: Path, out_dir: Path|None, topN_importances: int = 40):
    if not HAVE_SKLEARN: raise RuntimeError("scikit-learn + joblib are required.")
    payload = joblib.load(model_path)
    out_dir = _ensure_out(out_dir or (model_path.parent / "model_analysis"))
    model = payload.get("model"); classes = payload.get("classes", []); acc = payload.get("accuracy"); created_at = payload.get("created_at", "unknown"); train_time = payload.get("train_time_sec")
    X_train, y_train = payload.get("X_train"), payload.get("y_train")
    X_test, y_test = payload.get("X_test"), payload.get("y_test")
    y_pred, y_score = payload.get("y_pred"), payload.get("y_score")
    cm, cm_norm = payload.get("confusion_matrix"), payload.get("confusion_matrix_norm")
    per_class_metrics = payload.get("per_class_metrics", {})

    summary = {"accuracy": float(acc) if acc is not None else None, "created_at": created_at,
               "train_time_sec": float(train_time) if train_time is not None else None,
               "n_train": int(len(X_train)) if X_train is not None else 0,
               "n_test": int(len(X_test)) if X_test is not None else 0,
               "classes": classes, "topN_importances": topN_importances}
    with open(out_dir / "summary.json", "w") as f: json.dump(summary, f, indent=2)

    if cm is not None:
        plt.figure(figsize=(6,6)); ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(ax=plt.gca(), cmap="Blues", colorbar=False); plt.title("Confusion Matrix"); _savefig(out_dir/"confusion_matrix.png")
    if cm_norm is not None:
        plt.figure(figsize=(6,6)); ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=classes).plot(ax=plt.gca(), cmap="Blues", colorbar=False); plt.title("Normalized Confusion Matrix"); _savefig(out_dir/"confusion_matrix_normalized.png")

    if y_test is not None and y_pred is not None and len(classes)>0:
        headers = ["Class","Precision","Recall","F1","Support"]; rows=[]
        for cls in classes:
            m = per_class_metrics.get(cls, {})
            rows.append([cls, f"{m.get('precision',0):.3f}", f"{m.get('recall',0):.3f}", f"{m.get('f1',0):.3f}", str(m.get('support',0))])
        _table_image(headers, rows, "Per-class metrics", out_dir/"class_report_table.png")
        import csv
        with open(out_dir/"per_class_metrics.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow(headers)
            for r in rows: w.writerow(r)

    could_tsne = False
    if y_score is not None and y_test is not None and len(classes)>1:
        y_score = np.asarray(y_score); y_test_arr = np.asarray(y_test); y_bin = label_binarize(y_test_arr, classes=np.arange(len(classes)))
        plt.figure(figsize=(6,6))
        for i,cls in enumerate(classes):
            try:
                fpr,tpr,_ = roc_curve(y_bin[:,i], y_score[:,i]); roc_auc=auc(fpr,tpr); plt.plot(fpr,tpr,lw=2,label=f"{cls} (AUC={roc_auc:.2f})")
            except Exception: pass
        plt.plot([0,1],[0,1],"k--",lw=1); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curves (OvR)"); plt.legend(); _savefig(out_dir/"roc_curves.png")

        plt.figure(figsize=(6,6))
        for i,cls in enumerate(classes):
            try:
                precision,recall,_ = precision_recall_curve(y_bin[:,i], y_score[:,i]); ap = average_precision_score(y_bin[:,i], y_score[:,i]); plt.plot(recall,precision,lw=2,label=f"{cls} (AP={ap:.2f})")
            except Exception: pass
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curves (OvR)"); plt.legend(); _savefig(out_dir/"pr_curves.png")

        conf = np.max(y_score, axis=1)
        plt.figure(figsize=(6,4)); plt.hist(conf, bins=20, edgecolor="black", alpha=0.8); plt.title("Prediction Confidence Distribution (max P)"); plt.xlabel("Max predicted probability"); plt.ylabel("Frequency"); _savefig(out_dir/"confidence_histogram.png")

        correct = (np.argmax(y_score, axis=1) == y_test_arr).astype(int)
        bins = np.linspace(0.0,1.0,11); bin_ids = np.digitize(conf, bins)-1
        bin_acc, bin_conf, bin_counts = [], [], []
        for b in range(len(bins)-1):
            m = bin_ids==b
            if m.sum()>0:
                bin_acc.append(correct[m].mean()); bin_conf.append(conf[m].mean()); bin_counts.append(int(m.sum()))
        plt.figure(figsize=(6,6)); plt.plot([0,1],[0,1],"k--",lw=1,label="Perfect calibration"); plt.plot(bin_conf, bin_acc, marker="o", lw=2, label="Model")
        for x,y,n in zip(bin_conf, bin_acc, bin_counts): plt.text(x,y,str(n),fontsize=8,ha="center",va="bottom")
        plt.xlabel("Mean predicted probability (per bin)"); plt.ylabel("Empirical accuracy (per bin)"); plt.title("Reliability Diagram (Overall)"); plt.legend(); _savefig(out_dir/"reliability_diagram_overall.png")

    rf = None
    if model is not None:
        if isinstance(model, Pipeline):
            for _,step in model.named_steps.items():
                if hasattr(step,"feature_importances_"): rf = step; break
        if rf is None and hasattr(model,"feature_importances_"): rf = model

    if rf is not None and hasattr(rf,"feature_importances_"):
        importances = np.asarray(rf.feature_importances_)
        if importances.ndim==1:
            idx = np.argsort(-importances)[:topN_importances]; top_vals = importances[idx]; top_labels = [f"f{int(i)}" for i in idx]
            _barh(top_vals, top_labels, f"Random Forest Feature Importances (Top {len(idx)})", out_dir/"feature_importances_topN.png")

    if rf is not None and hasattr(rf,"estimators_"):
        depths, leaves = [], []
        try:
            for est in rf.estimators_:
                depths.append(est.tree_.max_depth); leaves.append(est.tree_.n_leaves)
        except Exception: pass
        if depths:
            fig, ax = plt.subplots(1,2,figsize=(9,3.5))
            ax[0].hist(depths, bins=min(20, max(5, int(np.sqrt(len(depths))))), edgecolor="black"); ax[0].set_title(f"Tree Depths (mean={np.mean(depths):.2f})"); ax[0].set_xlabel("Depth"); ax[0].set_ylabel("Count")
            ax[1].hist(leaves, bins=min(20, max(5, int(np.sqrt(len(leaves))))), edgecolor="black"); ax[1].set_title(f"#Leaves (mean={np.mean(leaves):.2f})"); ax[1].set_xlabel("Leaves"); ax[1].set_ylabel("Count")
            _savefig(out_dir/"rf_forest_stats.png")

    if per_class_metrics is not None and cm is not None and len(classes)>0:
        recalls = [per_class_metrics.get(cls,{}).get("recall",0.0) for cls in classes]; f1s = [per_class_metrics.get(cls,{}).get("f1",0.0) for cls in classes]
        _barh(recalls, classes, "Per-class Recall", out_dir/"recall_by_class.png"); _barh(f1s, classes, "Per-class F1", out_dir/"f1_by_class.png")
        pairs = _confusion_pairs(cm, classes)
        if pairs:
            import csv
            with open(out_dir/"confusion_pairs.csv","w",newline="") as f:
                w=csv.writer(f); w.writerow(["true","pred","count"])
                for t,p,c in pairs: w.writerow([t,p,c])
            top = pairs[:min(20,len(pairs))]; labels=[f"{t}→{p}" for t,p,_ in top]; counts=[c for _,_,c in top]
            _barh(counts, labels, "Top Confusions (off-diagonal CM counts)", out_dir/"top_confusions.png")
            top10 = pairs[:min(10,len(pairs))]; labels10=[f"{t}→{p}" for t,p,_ in top10]; counts10=[c for _,_,c in top10]
            _barh(counts10, labels10, "Top 10 Confusions", out_dir/"top_confusions_top10.png")

    could_tsne = False
    if X_test is not None and y_test is not None and len(X_test)>1:
        try:
            scaler = _safe_get_pipeline_step(model, StandardScaler)
            X_plot = scaler.transform(X_test) if scaler is not None else X_test
            N = len(X_plot); maxN = 2000
            if N>maxN:
                rng = np.random.default_rng(42); idx = rng.choice(N, size=maxN, replace=False); Xp = X_plot[idx]; yt = np.asarray(y_test)[idx]; yp = np.asarray(y_pred)[idx] if y_pred is not None else None
            else:
                Xp = X_plot; yt = np.asarray(y_test); yp = np.asarray(y_pred) if y_pred is not None else None
            tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, random_state=42, n_iter=1000)
            emb = tsne.fit_transform(Xp)
            plt.figure(figsize=(7,6))
            for i,cls in enumerate(classes):
                m = yt == i
                if m.sum()>0: plt.scatter(emb[m,0], emb[m,1], s=12, alpha=0.7, label=cls)
            if yp is not None:
                mis = yt != yp
                if mis.any(): plt.scatter(emb[mis,0], emb[mis,1], s=40, facecolors='none', edgecolors='k', linewidths=0.8, label="misclassified")
            plt.title("t-SNE (X_test) — colored by TRUE label"); plt.legend(markerscale=1.2, fontsize=8, framealpha=0.9); _savefig(out_dir/"tsne_test_true_vs_pred.png")
            could_tsne = True
        except Exception as e:
            warnings.warn(f"t-SNE failed: {e}")

    if y_score is not None and y_test is not None:
        y_score = np.asarray(y_score); y_test_arr = np.asarray(y_test); data, labels_box = [], []
        for i,cls in enumerate(classes):
            m = y_test_arr == i
            if m.sum()>0: data.append(y_score[m, i]); labels_box.append(cls)
        if data:
            plt.figure(figsize=(max(6, 0.35*len(labels_box)+3),4)); plt.boxplot(data, labels=labels_box, showfliers=False); plt.ylabel("Predicted probability of TRUE class"); plt.title("Per-class probability distribution (correct class prob)"); _savefig(out_dir/"per_class_prob_boxplots.png")

    artifacts = [
        "confusion_matrix.png","confusion_matrix_normalized.png","class_report_table.png","roc_curves.png","pr_curves.png",
        "confidence_histogram.png","reliability_diagram_overall.png","feature_importances_topN.png","rf_forest_stats.png",
        "recall_by_class.png","f1_by_class.png","top_confusions.png","top_confusions_top10.png",
        ("tsne_test_true_vs_pred.png" if could_tsne else "(tsne failed/skipped)"),
        "per_class_prob_boxplots.png","per_class_metrics.csv","confusion_pairs.csv","summary.json"
    ]
    with open(out_dir/"ARTIFACTS.txt","w") as f:
        for a in artifacts: f.write(f"{a}\n")
    print(f"✅ All analysis saved to: {out_dir.resolve()}")

def parse_args():
    p = argparse.ArgumentParser(description="Generate comprehensive model analysis PNGs for SmartTrashSorter")
    p.add_argument("--model", required=True, type=str)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--topN", type=int, default=40)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args(); model_path = Path(args.model); out_dir = Path(args.out) if args.out else None
    analyze(model_path, out_dir, topN_importances=args.topN)
