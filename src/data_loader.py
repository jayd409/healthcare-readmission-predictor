import pandas as pd
import numpy as np
import os

def load_data():
    """
    Generate realistic healthcare readmission predictor data based on CMS Hospital Readmission
    Reduction Program benchmarks.

    Overall 30-day readmission rate: ~15%
    High-risk conditions:
    - Heart Failure: ~24% readmission
    - COPD: ~19% readmission
    - Pneumonia: ~17% readmission
    - Hip/Knee Replacement: ~5% readmission
    - AMI (Acute MI): ~18% readmission

    Features: age, primary_diagnosis, los (length of stay), prior_admissions,
    comorbidities_count, discharge_disposition, insurance_type
    """
    os.makedirs('data', exist_ok=True)
    df = _generate_readmission_data(3000)
    df.to_csv('data/patient_data.csv', index=False)
    print(f"  Generated {len(df)} synthetic patient records (readmission prediction)")
    return df


def _generate_readmission_data(n=3000):
    """
    Generate realistic patient readmission data based on CMS benchmarks.
    """
    rng = np.random.default_rng(42)

    # Age distribution (readmission increases with age)
    age = rng.integers(18, 95, n)

    diagnoses = ['Heart Failure', 'COPD', 'Pneumonia', 'AMI', 'Hip/Knee Replacement',
                 'Stroke', 'Diabetes', 'Sepsis', 'Kidney Disease', 'Liver Disease']
    diagnosis_readmission_rates = {
        'Heart Failure': 0.24,
        'COPD': 0.19,
        'Pneumonia': 0.17,
        'AMI': 0.18,
        'Hip/Knee Replacement': 0.05,
        'Stroke': 0.12,
        'Diabetes': 0.08,
        'Sepsis': 0.22,
        'Kidney Disease': 0.15,
        'Liver Disease': 0.20
    }

    primary_diagnosis = rng.choice(diagnoses, n, p=[0.18, 0.15, 0.12, 0.10, 0.08, 0.08, 0.10, 0.07, 0.07, 0.05])

    # Length of stay (days) - typically 3-10 days, affected by diagnosis
    los = np.clip(rng.normal(5.5, 3, n), 1, 30).astype(int)

    prior_admissions = rng.integers(0, 6, n)

    comorbidities_count = rng.integers(0, 11, n)

    # Home (most common), Skilled Nursing Facility, Home Health Care, Rehab facility
    discharge_dispositions = ['Home', 'Skilled Nursing Facility', 'Home Health Care', 'Rehab', 'Other']
    disposition = rng.choice(discharge_dispositions, n, p=[0.50, 0.25, 0.15, 0.08, 0.02])

    insurance_types = ['Medicare', 'Medicaid', 'Commercial', 'Uninsured', 'Other']
    insurance_type = rng.choice(insurance_types, n, p=[0.45, 0.20, 0.25, 0.07, 0.03])

    # Sex (0=Female, 1=Male)
    sex = rng.integers(0, 2, n)

    # Generate readmission target based on realistic risk factors
    readmission_prob = np.zeros(n)

    for i in range(n):
        base_rate = diagnosis_readmission_rates[primary_diagnosis[i]]

        # Age factor (increases risk after 65)
        age_val = int(age[i]) if isinstance(age[i], np.integer) else age[i]
        age_factor = 1.0 + (age_val - 50) * 0.01 if age_val > 50 else 1.0
        age_factor = np.clip(age_factor, 0.8, 1.8)

        # Length of stay factor (longer stays = higher risk, indicator of severity)
        los_val = int(los[i]) if isinstance(los[i], np.integer) else los[i]
        los_factor = 1.0 + (los_val - 5) * 0.05
        los_factor = np.clip(los_factor, 0.7, 2.0)

        # Prior admissions factor (each prior admission increases risk)
        prior_val = int(prior_admissions[i]) if isinstance(prior_admissions[i], np.integer) else prior_admissions[i]
        prior_factor = 1.0 + (prior_val * 0.15)
        prior_factor = np.clip(prior_factor, 1.0, 2.5)

        # Comorbidities factor (each condition increases risk)
        comorbid_val = int(comorbidities_count[i]) if isinstance(comorbidities_count[i], np.integer) else comorbidities_count[i]
        comorbid_factor = 1.0 + (comorbid_val * 0.08)
        comorbid_factor = np.clip(comorbid_factor, 1.0, 2.0)

        # Disposition factor (SNF and rehab increase risk vs home)
        disposition_factor = {
            'Home': 1.0,
            'Skilled Nursing Facility': 1.5,
            'Home Health Care': 1.2,
            'Rehab': 1.1,
            'Other': 1.3
        }[disposition[i]]

        # Insurance factor (uninsured/Medicaid higher risk)
        insurance_factor = {
            'Medicare': 1.0,
            'Medicaid': 1.3,
            'Commercial': 0.9,
            'Uninsured': 1.4,
            'Other': 1.0
        }[insurance_type[i]]

        final_prob = base_rate * age_factor * los_factor * prior_factor * comorbid_factor * disposition_factor * insurance_factor
        readmission_prob[i] = np.clip(final_prob, 0.01, 0.9)

    target = (rng.random(n) < readmission_prob).astype(int)

    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'primary_diagnosis': primary_diagnosis,
        'los': los,
        'prior_admissions': prior_admissions,
        'comorbidities_count': comorbidities_count,
        'discharge_disposition': disposition,
        'insurance_type': insurance_type,
        'target': target,
    })

    # Encode categorical variables for model compatibility
    diagnosis_map = {d: i for i, d in enumerate(diagnoses)}
    df['primary_diagnosis_code'] = df['primary_diagnosis'].map(diagnosis_map)

    disposition_map = {d: i for i, d in enumerate(discharge_dispositions)}
    df['discharge_disposition_code'] = df['discharge_disposition'].map(disposition_map)

    insurance_map = {i: idx for idx, i in enumerate(insurance_types)}
    df['insurance_code'] = df['insurance_type'].map(insurance_map)

    # Drop original categorical columns and keep the encoded versions for model training
    df = df.drop(['primary_diagnosis', 'discharge_disposition', 'insurance_type'], axis=1)
    df = df.rename(columns={
        'primary_diagnosis_code': 'primary_diagnosis',
        'discharge_disposition_code': 'discharge_disposition',
        'insurance_code': 'insurance_type'
    })

    return df
