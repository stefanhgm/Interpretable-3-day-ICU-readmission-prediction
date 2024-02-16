# Code for *Development and Validation of an Interpretable 3-day Intensive Care Unit Readmission Prediction Model Using Explainable Boosting Machines*

This repository contains the code used for all experiments described in *Development and Validation of an Interpretable 3-day Intensive Care Unit Readmission Prediction Model Using Explainable Boosting Machines*.

Please consider citing:

```
@article{hegselmann2022development,
  title={Development and validation of an interpretable 3 day intensive care unit readmission prediction model using explainable boosting machines},
  author={Hegselmann, Stefan and Ertmer, Christian and Volkert, Thomas and Gottschalk, Antje and Dugas, Martin and Varghese, Julian},
  journal={Frontiers in Medicine},
  volume={9},
  pages={960296},
  year={2022},
  publisher={Frontiers}
}
```

## Project structure

* **`data_description`** 
Scripts to describe the data. Used to generate tables in the manuscript.

* **`experiments`**
Main code to conduct experiments.

* **`helper`**
Utility function used across the project.

* **`preprocessing`**
Code for data preprocessing of the UKM cohort.

* **`preprocessing_mimic`**
Code for data preprocessing of the MIMIC-IV cohort.

* **`research_database`**
SQL statements to create research database and functionality to interact with it.
