================================================================================

ENERGY CONSUMPTION & LOAD PREDICTION - PROJECT DOCUMENTATION

================================================================================

1. PROJECT OVERVIEW

--------------------------------------------------------------------------------

This project implements an end-to-end Machine Learning pipeline in Python to

analyze industrial energy data, handle missing records, extract time-based

features, and build a robust predictive model for electricity usage (Usage_kWh).

2. DIRECTORY STRUCTURE

--------------------------------------------------------------------------------

UX_Project/

├── load_data.csv          # Raw input dataset

│

├── cleaned_load_data.csv          # Processed and encoded dataset (output)

├── main.py                        # Complete end-to-end Python pipeline script

└── README.txt                     # Project documentation text file

3. PREREQUISITES & INSTALLATION

--------------------------------------------------------------------------------

Ensure you have Python installed, then install the required data science packages

by running the following command in your terminal:

pip install pandas scikit-learn

4. STEP-BY-STEP PROCEDURE TO RUN THE PROJECT

--------------------------------------------------------------------------------

Step 1: Set up your folder structure and place your dataset inside:

data/ux data/load_data.csv

Step 2: Ensure your `main.py` script contains the complete end-to-end code.

Step 3: Open your terminal or command prompt, navigate to your project directory,

and execute the script:

python main.py

5. PIPELINE METHODOLOGY & ARCHITECTURE

--------------------------------------------------------------------------------

A. Data Ingestion & Inspection:

- Loads the raw CSV file into a Pandas DataFrame.

- Inspects the initial dimensions (35,041 rows, 9 columns) and data hygiene.

B. Data Cleaning & Imputation:

- Identifies missing values across numerical attributes.

- Imputes missing records using column medians to prevent data skewness.

C. Feature Engineering & Categorical Encoding:

- Parses the raw timestamp column (`Date_Time`) into true datetime objects.

- Extracts cyclical temporal features: `Month`, `Hour`, and `DayOfWeek`.

- Applies One-Hot Encoding to categorical variables like `Load_Type`

(converting text labels into boolean indicators).

- Exports the fully cleaned dataset to `cleaned_load_data.csv`.

D. Machine Learning Model Training & Evaluation:

- Splits the processed dataset into 80% training and 20% testing sets.

- Trains a powerful ensemble model: **Random Forest Regressor**.

- Evaluates performance using Mean Squared Error (MSE) and R-squared ($R^2$).

6. MODEL PERFORMANCE RESULTS

--------------------------------------------------------------------------------

- Algorithm Used    : Random Forest Regressor

- Mean Squared Error: 417.7571

- R-squared Score   : 0.7486 (Indicates ~75% variance explained)

================================================================================
