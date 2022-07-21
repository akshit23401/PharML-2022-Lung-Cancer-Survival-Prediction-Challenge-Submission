# PharML-2022-Lung-Cancer-Survival-Prediction-Challenge-Submission
My Submission for the PharML 2022 Lung Cancer Survival Prediction Challenge organized by ECML PKDD 2022.

  Description of the challenge:  Every year 1.9 million people are diagnosed with Non-Small Cell Lung Cancer (NSCLC). Only 25% of those will survive 5 years beyond their diagnosis, with prognosis depending on many factors including demographics, clinical characteristics, and genetic alterations, among others.   Survival Machine Learning models could enable us to better predict prognosis for individual patients, which in turn has real-world clinical applications for improving treatment and our understanding of NSCLC.
  
Additionally, representations learned by fitting Survival Machine Learning models on NSCLC data could be used to stratify patients and obtain clinical clusters or phenotypes that give us insights on how to better categorize NSCLCs.

In this challenge, you will predict the risk of overall death using clinical EHR data from around 75,000 advanced NSCLC patients provided by Flatiron Health. The features consist of patient characteristics such as demographic information, vital sign data, and biomarkers.

I have implemented Survival Machine Learning models to predict the risk of death for patients with Non-Small Cell Lung Cancer.
Trained various models on dataset of 32000 patients with 25 features for each patient to predict the number of days before their demise based on health conditions and other features.

Attained concordance index of 0.58 on the test dataset using CatBoost algorithm and various optimization techniques
