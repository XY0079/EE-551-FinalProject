"""Energy analyzer for CSV and UCI TXT datasets."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class EnergyAnalyzer:
    """Analyze energy time-series data."""
    csv_path: Optional[str] = None

    def load(self) -> pd.DataFrame:
        """Load dataset from csv_path."""
        if not self.csv_path:
            raise ValueError("csv_path is not provided. Set EnergyAnalyzer(csv_path=...) or pass a DataFrame directly.")
        p = Path(self.csv_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {self.csv_path}")

        # Detect format by extension or header
        if p.suffix.lower() == ".txt":
            df = self._load_uci_txt(p)
        else:
            df = self._load_csv_generic(p)

        # Final sanity checks
        if 'date' not in df.columns:
            raise ValueError("Loaded data must contain a 'date' column")
        if 'energy_kwh' not in df.columns:
            raise ValueError("Loaded data must contain an 'energy_kwh' column")
        return df

    def _load_csv_generic(self, path: Path) -> pd.DataFrame:
        """Load CSV file."""
        df = None
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {e}") from e
        finally:
            if df is None:
                pass
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
        else:
            raise ValueError("CSV must contain 'timestamp' or 'date' column")

        if 'energy_kwh' not in df.columns:
            if 'power_w' in df.columns:
                minutes = df['minutes'] if 'minutes' in df.columns else 60.0
                df['energy_kwh'] = (df['power_w'] * (minutes/60.0)) / 1000.0
            else:
                raise ValueError("CSV must have 'energy_kwh' or 'power_w' (+ optional 'minutes')")
        return df

    def _load_uci_txt(self, path: Path) -> pd.DataFrame:
        """Load UCI TXT file (semicolon-separated)."""
        try:
            df = pd.read_csv(
                path,
                sep=';',
                parse_dates={'datetime': ['Date', 'Time']},
                na_values='?',
                low_memory=False,
            )
        except Exception as e:
            raise ValueError(f"Failed to read UCI TXT: {e}") from e

        required = [
            'datetime','Global_active_power','Voltage','Global_intensity',
            'Sub_metering_1','Sub_metering_2','Sub_metering_3'
        ]
        df = df.dropna(subset=required)

        num_cols = ['Global_active_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=num_cols)

        df['energy_kwh'] = df['Global_active_power'] / 60.0
        df['date'] = df['datetime'].dt.date
        return df

    def daily_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate daily energy usage."""
        if not {'date','energy_kwh'} <= set(df.columns):
            raise ValueError("DataFrame must contain 'date' and 'energy_kwh'")
        daily = df.groupby('date', as_index=False)['energy_kwh'].sum()
        daily.rename(columns={'energy_kwh':'daily_kwh'}, inplace=True)
        return daily

    def plot_daily(self, daily_df: pd.DataFrame) -> None:
        """Plot daily energy usage."""
        plt.figure()
        plt.plot(pd.to_datetime(daily_df['date']), daily_df['daily_kwh'])
        plt.xlabel('Date')
        plt.ylabel('Energy (kWh)')
        plt.title('Daily Energy Consumption')
        plt.grid(True)
        plt.tight_layout()

    def predict_linear(self, daily_df: pd.DataFrame, days_ahead: int = 7) -> pd.DataFrame:
        """Predict future energy using linear regression."""
        x = np.arange(len(daily_df), dtype=float)
        y = daily_df['daily_kwh'].to_numpy(dtype=float)
        if len(x) < 2:
            raise ValueError("Need at least two days of data for linear prediction")
        coef = np.polyfit(x, y, 1)
        model = np.poly1d(coef)

        future_x = np.arange(len(daily_df), len(daily_df) + days_ahead, dtype=float)
        preds = model(future_x)

        last_date = pd.to_datetime(daily_df['date']).max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days_ahead, freq='D').date
        return pd.DataFrame({'date': future_dates, 'pred_kwh': preds})

    def validate_data_quality(self, df: pd.DataFrame, max_retries: int = 3) -> pd.DataFrame:
        """Validate and clean data."""
        retries = 0
        cleaned_df = df.copy()
        
        while retries < max_retries:
            invalid_indices = [
                idx for idx, row in cleaned_df.iterrows()
                if pd.isna(row.get('energy_kwh', None)) or row.get('energy_kwh', 0) < 0
            ]
            
            if not invalid_indices:
                break
                
            is_valid = lambda row: not pd.isna(row.get('energy_kwh', None)) and row.get('energy_kwh', 0) >= 0
            valid_mask = cleaned_df.apply(is_valid, axis=1)
            cleaned_df = cleaned_df[valid_mask].reset_index(drop=True)
            
            retries += 1
        
        return cleaned_df

    def generate_daily_stats(self, daily_df: pd.DataFrame) -> Generator[dict, None, None]:
        """Generate daily statistics."""
        for _, row in daily_df.iterrows():
            stats = {
                'date': row['date'],
                'daily_kwh': row['daily_kwh'],
                'timestamp': datetime.now().isoformat()
            }
            yield stats

    def aggregate_by_month(self, daily_df: pd.DataFrame) -> dict:
        """Aggregate daily data by month."""
        monthly_totals = defaultdict(float)
        
        for _, row in daily_df.iterrows():
            date = pd.to_datetime(row['date'])
            month_key = f"{date.year}-{date.month:02d}"
            monthly_totals[month_key] += row['daily_kwh']
        
        return {month: round(total, 2) for month, total in monthly_totals.items()}


if __name__ == "__main__":
    print("EnergyAnalyzer module loaded successfully.")
    print("This module provides classes and functions for energy data analysis.")
    print("Import this module in your Jupyter notebook or other Python scripts.")
