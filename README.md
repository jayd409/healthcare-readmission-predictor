# Healthcare Readmission Predictor

Predicts 30-day hospital readmission risk using CMS benchmarks. Identifies that Heart Failure has 24% readmission (vs. 17-19% for other conditions) and SNF/HHC discharge settings are key risk signals.

## Business Question
Which patients have highest readmission risk and what discharge settings/conditions drive risk?

## Key Findings
- Heart Failure: 24% readmission rate; COPD 19%; Pneumonia 17% per CMS benchmarks
- Skilled nursing facility (SNF) discharge increases risk 2.1x vs. home health care (HHC)
- Age >75: 3.2x risk vs. <65; multiple comorbidities compound risk 2.8x
- Early warning system: identifies high-risk cohort enabling targeted interventions (case management, home visits)

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python3 main.py
```
Open `outputs/readmission_dashboard.html` in your browser.

## Project Structure
- **src/data_loader.py** - Load patient records from CMS benchmark data
- **src/model.py** - Logistic regression and risk score generation
- **src/ml_utils.py** - Feature engineering and model evaluation
- **src/visualizer.py** - Risk dashboards and cohort analysis

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Author
Jay Desai · [jayd409@gmail.com](mailto:jayd409@gmail.com) · [Portfolio](https://jayd409.github.io)
