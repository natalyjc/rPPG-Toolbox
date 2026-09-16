import numpy as np
import pandas as pd
import torch
import os
import json
import pickle 
from evaluation.post_process import *
from evaluation.BlandAltmanPy import BlandAltman

def calculate_few_shot_metrics(
    gt_hr_all, baseline_hr_all, adapted_hr_all,
    baseline_snr_all, adapted_snr_all
):
    """Summarize, print, and return baseline/adapted few-shot metrics."""
    baseline_metrics = summarize_window_metrics(baseline_hr_all, gt_hr_all, baseline_snr_all)
    adapted_metrics = summarize_window_metrics(adapted_hr_all, gt_hr_all, adapted_snr_all)
    return baseline_metrics, adapted_metrics

def summarize_window_metrics(pred_hr_all, gt_hr_all, snr_all=None):
    pred_hr_all = np.asarray(pred_hr_all)
    gt_hr_all = np.asarray(gt_hr_all)
    differences = pred_hr_all - gt_hr_all
    absolute_differences = np.abs(differences)
    rho = float(np.corrcoef(pred_hr_all, gt_hr_all)[0, 1]) if len(pred_hr_all) > 1 else float('nan')
    summary = {
        "MAE": float(np.mean(absolute_differences)),
        "RMSE": float(np.sqrt(np.mean(np.square(differences)))),
        "Rho": rho,
        "MAPE": float(np.mean(absolute_differences / np.clip(np.abs(gt_hr_all), 1e-8, None)) * 100.0),
    }
    if snr_all is not None:
        snr_all = np.asarray(snr_all)
        summary["SNR"] = float(np.mean(snr_all)) if snr_all.size else float('nan')
    return summary

def save_few_shot_plots(gt_hr_all, baseline_hr_all, adapted_hr_all, config):
    """Save baseline and adapted few-shot Bland-Altman plots."""
    if not config.TEST.OUTPUT_SAVE_DIR:
        return

    baseline_compare = BlandAltman(gt_hr_all, baseline_hr_all, config, averaged=True)
    adapted_compare = BlandAltman(gt_hr_all, adapted_hr_all, config, averaged=True)
    plot_args = {
        "x_label": 'GT PPG HR [bpm]',
        "y_label": 'rPPG HR [bpm]',
        "show_legend": True,
        "figure_size": (5, 5),
    }
    for compare, tag in ((baseline_compare, "baseline"), (adapted_compare, "adapted")):
        prefix = f"PURE_RhythmFormer_UBFC-rPPG_fewshot_{tag}_BlandAltman"
        compare.scatter_plot(
            **plot_args,
            the_title=f"{prefix}_ScatterPlot",
            file_name=f"{prefix}_ScatterPlot.pdf",
        )
        compare.difference_plot(
            x_label='Difference between rPPG HR and GT PPG HR [bpm]',
            y_label='Average of rPPG HR and GT PPG HR [bpm]',
            show_legend=True,
            figure_size=(5, 5),
            the_title=f"{prefix}_DifferencePlot",
            file_name=f"{prefix}_DifferencePlot.pdf",
        )
def save_few_shot_reports(results, baseline_predictions, adapted_predictions, labels, config):
    """Save few-shot pickle, JSON, and CSV reports."""
    if not config.TEST.OUTPUT_SAVE_DIR:
        return

    baseline_metrics = results["baseline"]
    adapted_metrics = results["adapted"]

    per_subject_metrics = results["per_subject_metrics"]
    subject_query_window_counts = results["subject_query_window_counts"]

    baseline_pickle_metrics = {
        "overall": baseline_metrics,
        "per_subject": {subject: metrics["baseline"] for subject, metrics in per_subject_metrics.items()},
        "subject_query_window_counts": subject_query_window_counts,
    }
    adapted_pickle_metrics = {
        "overall": adapted_metrics,
        "per_subject": {subject: metrics["adapted"] for subject, metrics in per_subject_metrics.items()},
        "subject_query_window_counts": subject_query_window_counts,
    }

    save_few_shot_outputs(baseline_predictions, labels, config, tag="baseline", metrics=baseline_pickle_metrics)
    save_few_shot_outputs(adapted_predictions, labels, config, tag="adapted", metrics=adapted_pickle_metrics)

    results_path = os.path.join(config.TEST.OUTPUT_SAVE_DIR, 'few_shot_rhythmformer_results.json')
    with open(results_path, 'w') as file:
        json.dump(results, file, indent=2)
    print(f"Saved few-shot summary to: {results_path}")

    overall = {
        "support_frames": results["support_frames"],
        "adapt_steps": results["adapt_steps"],
        "window_frames": results["window_frames"],
        "num_subjects": results["num_subjects"]
    }
    for tag, metrics in (("baseline", baseline_metrics), ("adapted", adapted_metrics)):
        for metric in ("MAE", "RMSE", "MAPE", "Rho", "SNR"):
            overall[f"{tag}_{metric}"] = metrics[metric]
    pd.DataFrame([overall]).to_csv(os.path.join(config.TEST.OUTPUT_SAVE_DIR, "overall_metrics.csv"), index=False)

    rows = []
    for subject, metrics in per_subject_metrics.items():
        row = {"subject": subject, "query_window_count": metrics["query_window_count"]}
        for tag in ("baseline", "adapted"):
            for metric in ("MAE", "RMSE", "MAPE", "Rho", "SNR"):
                row[f"{tag}_{metric}"] = metrics[tag][metric]
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(config.TEST.OUTPUT_SAVE_DIR, "per_subject_metrics.csv"), index=False)

def save_few_shot_outputs(predictions, labels, config, tag, metrics):
    """Save few-shot frame-level predictions and labels to pickle, mirroring save_test_outputs."""
    output_dir = config.TEST.OUTPUT_SAVE_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Filename ID to be used in any output files that get saved
    if config.TOOLBOX_MODE == 'only_test':
        model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
        filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
    else:
        raise ValueError('few-shot output saving only supports only_test!')
    output_path = os.path.join(output_dir, f"{filename_id}_fewshot_{tag}_outputs.pickle")

    data = dict()
    # In _save_few_shot_outputs, instead of storing raw arrays, wrap in dict with a single chunk key
    data['predictions'] = {subj: {0: torch.tensor(arr)} for subj, arr in predictions.items()}
    data['labels'] = {subj: {0: torch.tensor(arr)} for subj, arr in labels.items()}
    data['metrics'] = metrics
    data['label_type'] = config.TEST.DATA.PREPROCESS.LABEL_TYPE
    data['fs'] = config.TEST.DATA.FS

    with open(output_path, 'wb') as handle:  # save out frame dict pickle file
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Saving few-shot {tag} outputs to: {output_path}')

