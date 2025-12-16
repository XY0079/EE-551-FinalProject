# Appliance Energy Consumption Analyzer & Predictor

## Project Title
**Appliance Energy Consumption Analyzer & Predictor**

## Students (Team Members)
- **Tianyi Lu** — XY0079
- **Kuo Yu** -- yukuo31415-web

## Problem Description
This project analyzes and forecasts household/appliance electricity consumption using Python. It loads the UCI “Individual household electric power consumption” dataset (or a compatible CSV), aggregates energy usage (daily/monthly), visualizes trends, and performs a simple linear forecast for upcoming days.

## Repository / Program Structure
```
code/
├── main_uci.ipynb                  # Main program (Jupyter Notebook)
├── analyzer.py                     # EnergyAnalyzer: load/clean/aggregate/plot/predict
├── appliance.py                    # Appliance + ApplianceList (composition, __str__, __add__)
├── test_analyzer.py                # Pytest tests
├── household_power_consumption.txt # UCI TXT dataset (example)
├── daily_ucidata.csv               # Output: daily aggregation
└── pred_ucidata.csv                # Output: forecast results
```

## How to Run
### Requirements
- Python **3.12** or **3.13**
- Packages: `pandas`, `numpy`, `matplotlib`, `pytest`

### Install
```bash
pip install pandas numpy matplotlib pytest
```

### Run the main notebook
1. Open the notebook:
```bash
jupyter notebook code/main_uci.ipynb
```
2. In `main_uci.ipynb`, set `TXT_PATH` to your local UCI TXT file path (or keep the provided example if the file exists).
3. Run all cells. Outputs will be saved as `code/daily_ucidata.csv` and `code/pred_ucidata.csv`.

### Run tests
```bash
pytest -v code/test_analyzer.py
```

## Main Contributions (Equal Split)
### Tianyi Lu
- Implemented `EnergyAnalyzer` (`analyzer.py`): data loading/cleaning, daily & monthly aggregation, plotting, linear forecasting, and generator-based stats.
- Built the end-to-end workflow in `main_uci.ipynb` (load → clean → aggregate → plot → forecast → export).
- Maintained documentation (README + docstrings/comments).

### Kuo Yu
- Implemented appliance modeling (`appliance.py`): `Appliance`, `ApplianceList`, input validation, `__str__`, and operator overloading (`__add__`).
- Wrote and maintained the Pytest suite (`test_analyzer.py`) covering core functionality and exception cases.
- Performed code quality checks and debugging to ensure the project runs correctly.

## Notes
- The UCI TXT file is large; first run may take some time depending on your machine.
- Ensure `TXT_PATH` in the notebook points to a valid file on your computer.
