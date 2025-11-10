import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import save_dashboard

def build_dashboard(df_scored, metrics, feature_importance, test_data):
    """Build 6-chart healthcare prediction dashboard."""

    sns.set_style("whitegrid")

    # Chart 1: Readmission/disease rate by age group
    fig, ax = plt.subplots(figsize=(10, 5))
    df_scored['age_group'] = pd.cut(df_scored['age'], bins=[0,40,50,60,70,80], labels=['<40','40-50','50-60','60-70','70+'])
    age_disease = df_scored.groupby('age_group', observed=True)['target'].agg(['sum', 'count'])
    age_disease['rate'] = (age_disease['sum'] / age_disease['count']).round(3)
    ax.bar(range(len(age_disease)), age_disease['rate'], color='steelblue')
    ax.set_xticks(range(len(age_disease)))
    ax.set_xticklabels(age_disease.index)
    ax.set_ylabel('Disease Rate')
    ax.set_xlabel('Age Group')
    ax.set_title('Disease Prevalence by Age', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    for i, v in enumerate(age_disease['rate']):
        ax.text(i, v+0.03, f'{v:.1%}', ha='center')
    charts = {'Disease by Age': fig}

    # Chart 2: Feature importance (top 10)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = feature_importance.head(10)
    ax.barh(range(len(top_features)), top_features['importance'].values, color='teal')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('Importance')
    ax.set_title('Top 10 Predictive Features', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    charts['Feature Importance'] = fig

    # Chart 3: Risk score distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df_scored['risk_score'], bins=40, color='coral', alpha=0.7, edgecolor='black')
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    ax.set_xlabel('Predicted Risk Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Risk Score Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    charts['Risk Distribution'] = fig

    # Chart 4: Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    conf_mat = np.array([
        [metrics['tn'], metrics['fp']],
        [metrics['fn'], metrics['tp']]
    ])
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'],
                cbar=False)
    ax.set_title('Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    charts['Confusion Matrix'] = fig

    # Chart 5: ROC-like curve (TPR vs FPR)
    fig, ax = plt.subplots(figsize=(8, 6))
    X_test, y_test, proba_test = test_data
    thresholds = np.linspace(0, 1, 50)
    tprs, fprs = [], []
    for t in thresholds:
        pred = (proba_test >= t).astype(int)
        tp = ((pred == 1) & (y_test == 1)).sum()
        fp = ((pred == 1) & (y_test == 0)).sum()
        tn = ((pred == 0) & (y_test == 0)).sum()
        fn = ((pred == 0) & (y_test == 1)).sum()
        tprs.append(tp / max(tp + fn, 1))
        fprs.append(fp / max(fp + tn, 1))
    ax.plot(fprs, tprs, 'b-', linewidth=2, label=f'AUC = {metrics["auc"]:.3f}')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    charts['ROC Curve'] = fig

    # Chart 6: Patient gender disease prevalence
    fig, ax = plt.subplots(figsize=(10, 5))
    gender_disease = df_scored.groupby('sex')['target'].agg(['sum', 'count'])
    gender_disease['rate'] = (gender_disease['sum'] / gender_disease['count']).round(3)
    gender_labels = ['Female', 'Male']
    ax.bar(gender_labels, gender_disease['rate'].values, color=['pink', 'lightblue'])
    ax.set_ylabel('Disease Rate')
    ax.set_title('Disease Prevalence by Gender', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    for i, v in enumerate(gender_disease['rate'].values):
        ax.text(i, v+0.03, f'{v:.1%}', ha='center')
    charts['Disease by Gender'] = fig

    # KPIs
    high_risk = (df_scored['risk_score'] >= 0.5).sum()
    kpis = {
        'Model Accuracy': f"{metrics['test_accuracy']:.1%}",
        'AUC Score': f"{metrics['auc']:.3f}",
        'High-Risk Patients': f"{high_risk:,}",
        'Sensitivity': f"{metrics['sensitivity']:.1%}",
    }

    os.makedirs('outputs', exist_ok=True)
    save_dashboard(charts, 'Healthcare Readmission Predictor', 'outputs/dashboard.html', kpis=kpis)
