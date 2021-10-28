# learning_alignment
Code associated with alignment experiment paper. Data is hosted on OSF, and can be accessed via <a href="https://osf.io/v8adf/?view_only=12edade933a145249cf6db17428ae474">this link</a>.

## Repo structure

```
learning_alignment
├── code
│   ├── processing
│   ├── analysis
│   └── modelling
├── data
│   ├── experiment
│   │   ├── processed
│   │   └── raw
│   └── modelling
│   │   ├── new_fits
│   │   └── paper_results
├── docs
└── README.md
```

## Repo usage
A rough pipeline for usage of this repo is as follows:

**To replicate analyses of collected data:**
1. Clone repo to local machine
2. Save data from OSF to `./data/experiment/raw`
3. Run `./code/processing/processing_script.py` to generate processed data files and replicate plots
4. Run `./code/analysis/analysis_script.R` to replicate statistical analyses

**To replicate model fitting procedure:**
1. Clone repo to local machine
2. Save data from OSF to `./data/experiment/raw`
3. Change script `./code/modelling/model_fitting_script.py` setting `plot_fitted` to False on line 149
4. Run above script
  - Note that this simple script runs fits for participants in series. If seeking to run this for all participants, this would take a very long time. If you wanted to replicate the full procedure, we recommend splitting the participant set into parallel streams and running across multiple CPUs. 

**To replicate plots for paper model fits:**
1. Run `./code/modelling/model_fitting_script.py`
  - Fitted hyperparameters from paper are saved in `./data/models/paper_results/fitted_model_hyperparams.csv`

