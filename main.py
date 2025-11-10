#!/usr/bin/env python3
import sys, os
sys.path.insert(0, 'src')

from data_loader import load_data
from model import train_disease_predictor
from visualizer import build_dashboard
from database import save_to_db, query

def main():
    print("Healthcare Readmission Predictor")
    print("=" * 50)

    print("\nLoading patient data...")
    df = load_data()

    print("Training disease prediction model...")
    metrics, df_scored, feature_importance, test_data = train_disease_predictor(df)

    # Print model performance
    print(f"\n  Model Performance  :")
    print(f"    Accuracy        : {metrics['test_accuracy']:.1%}")
    print(f"    AUC Score       : {metrics['auc']:.3f}")
    print(f"    Sensitivity     : {metrics['sensitivity']:.1%}")
    print(f"    Specificity     : {metrics['specificity']:.1%}")

    high_risk = (df_scored['risk_score'] >= 0.5).sum()
    print(f"\n  Cohort Analysis    :")
    print(f"    Total Patients  : {len(df):,}")
    print(f"    High-Risk       : {high_risk:,} ({high_risk/len(df):.1%})")
    print(f"    Disease Present : {df['target'].sum():,} ({df['target'].mean():.1%})")

    print("\nBuilding dashboard...")
    build_dashboard(df_scored, metrics, feature_importance, test_data)

    # Save scored dataset
    df_scored.to_csv('outputs/patient_predictions.csv', index=False)
    print(f"  Patient predictions → outputs/patient_predictions.csv")

    print("\n--- SQLite Storage ---")
    save_to_db(df_scored, 'patients')

    # SQL Analytics (SQLite)
    print("\n--- SQL Analytics (SQLite) ---")

    # 1. Readmission/target rate by age group
    print("\n1. Disease Rate by Age Group:")
    result = query("""
        SELECT
               CASE
                   WHEN age < 40 THEN '30-39'
                   WHEN age < 50 THEN '40-49'
                   WHEN age < 60 THEN '50-59'
                   WHEN age < 70 THEN '60-69'
                   ELSE '70+'
               END as age_group,
               COUNT(*) as patient_count,
               SUM(target) as disease_count,
               ROUND(AVG(target) * 100, 1) as disease_rate_pct,
               ROUND(AVG(risk_score) * 100, 1) as avg_risk_score_pct
        FROM patients
        GROUP BY age_group
        ORDER BY age_group
    """)
    print(result.to_string(index=False))

    # 2. Risk score distribution
    print("\n2. Risk Score Distribution:")
    result = query("""
        SELECT
               CASE
                   WHEN risk_score < 0.33 THEN 'Low (0-0.33)'
                   WHEN risk_score < 0.67 THEN 'Medium (0.33-0.67)'
                   ELSE 'High (0.67-1.0)'
               END as risk_tier,
               COUNT(*) as patient_count,
               ROUND(AVG(risk_score), 3) as avg_risk_score,
               SUM(target) as disease_count
        FROM patients
        GROUP BY risk_tier
        ORDER BY risk_tier
    """)
    print(result.to_string(index=False))

    # 3. High risk patient count summary
    print("\n3. High-Risk Patient Summary:")
    result = query("""
        SELECT
               CASE WHEN risk_score >= 0.5 THEN 'High Risk' ELSE 'Lower Risk' END as risk_category,
               COUNT(*) as patient_count,
               SUM(target) as disease_count,
               ROUND(AVG(target) * 100, 1) as disease_rate_pct
        FROM patients
        GROUP BY risk_category
        ORDER BY risk_category DESC
    """)
    print(result.to_string(index=False))

    print("\nDone. Open outputs/dashboard.html to view.")

if __name__ == "__main__":
    main()
