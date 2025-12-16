"""Tests for EnergyAnalyzer and Appliance modules."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from analyzer import EnergyAnalyzer
from appliance import Appliance, ApplianceList


class TestEnergyAnalyzer:
    """Tests for EnergyAnalyzer."""
    
    def test_load_csv_generic_success(self):
        """Test CSV file loading."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("timestamp,energy_kwh\n")
            f.write("2024-01-01 00:00:00,10.5\n")
            f.write("2024-01-01 01:00:00,12.3\n")
            f.write("2024-01-02 00:00:00,11.2\n")
            temp_path = f.name
        
        try:
            analyzer = EnergyAnalyzer(csv_path=temp_path)
            df = analyzer.load()
            
            # Verify the loaded data
            assert 'date' in df.columns
            assert 'energy_kwh' in df.columns
            assert len(df) == 3
            assert df['energy_kwh'].sum() == pytest.approx(34.0, rel=1e-2)
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_load_file_not_found_exception(self):
        """Test FileNotFoundError handling."""
        analyzer = EnergyAnalyzer(csv_path="nonexistent_file.csv")
        
        with pytest.raises(FileNotFoundError):
            analyzer.load()
    
    def test_daily_usage_aggregation(self):
        """Test daily aggregation."""
        # Create sample data with multiple entries per day
        data = {
            'date': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02']).date,
            'energy_kwh': [10.0, 15.0, 12.0, 18.0]
        }
        df = pd.DataFrame(data)
        
        analyzer = EnergyAnalyzer()
        daily = analyzer.daily_usage(df)
        
        # Verify aggregation
        assert len(daily) == 2
        assert 'daily_kwh' in daily.columns
        assert daily[daily['date'] == pd.to_datetime('2024-01-01').date()]['daily_kwh'].values[0] == 25.0
        assert daily[daily['date'] == pd.to_datetime('2024-01-02').date()]['daily_kwh'].values[0] == 30.0
    
    def test_predict_linear_insufficient_data(self):
        """Test ValueError for insufficient data."""
        # Create data with only one day
        data = {
            'date': [pd.to_datetime('2024-01-01').date()],
            'daily_kwh': [10.0]
        }
        daily_df = pd.DataFrame(data)
        
        analyzer = EnergyAnalyzer()
        
        with pytest.raises(ValueError, match="at least two days"):
            analyzer.predict_linear(daily_df, days_ahead=7)
    
    def test_validate_data_quality(self):
        """Test data validation."""
        # Create data with some invalid entries
        data = {
            'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']).date,
            'energy_kwh': [10.0, -5.0, np.nan]  # Invalid: negative and NaN
        }
        df = pd.DataFrame(data)
        
        analyzer = EnergyAnalyzer()
        cleaned_df = analyzer.validate_data_quality(df)
        
        # Verify invalid rows are removed
        assert len(cleaned_df) == 1
        assert cleaned_df['energy_kwh'].iloc[0] == 10.0
    
    def test_generate_daily_stats_generator(self):
        """Test generator function."""
        data = {
            'date': pd.to_datetime(['2024-01-01', '2024-01-02']).date,
            'daily_kwh': [10.0, 15.0]
        }
        daily_df = pd.DataFrame(data)
        
        analyzer = EnergyAnalyzer()
        stats_gen = analyzer.generate_daily_stats(daily_df)
        
        # Verify generator yields correct data
        stats_list = list(stats_gen)
        assert len(stats_list) == 2
        assert stats_list[0]['date'] == pd.to_datetime('2024-01-01').date()
        assert stats_list[0]['daily_kwh'] == 10.0
        assert 'timestamp' in stats_list[0]
    
    def test_aggregate_by_month(self):
        """Test monthly aggregation."""
        data = {
            'date': pd.to_datetime(['2024-01-01', '2024-01-15', '2024-02-01']).date,
            'daily_kwh': [10.0, 15.0, 20.0]
        }
        daily_df = pd.DataFrame(data)
        
        analyzer = EnergyAnalyzer()
        monthly = analyzer.aggregate_by_month(daily_df)
        
        # Verify monthly aggregation
        assert '2024-01' in monthly
        assert '2024-02' in monthly
        assert monthly['2024-01'] == 25.0
        assert monthly['2024-02'] == 20.0


class TestAppliance:
    """Tests for Appliance and ApplianceList."""
    
    def test_appliance_creation(self):
        """Test Appliance creation."""
        app = Appliance("Microwave", 1200.0, 10.0)
        
        assert app.name == "Microwave"
        assert app.power_w == 1200.0
        assert app.usage_minutes == 10.0
    
    def test_appliance_energy_calculation(self):
        """Test energy calculation."""
        app = Appliance("Kettle", 2000.0, 5.0)
        energy = app.energy_kwh_single_use()
        
        expected = 2000.0 * (5.0 / 60.0) / 1000.0
        assert energy == pytest.approx(expected, rel=1e-3)
    
    def test_appliance_str_representation(self):
        """Test __str__ method."""
        app = Appliance("Refrigerator", 150.0, 1440.0)
        str_repr = str(app)
        
        assert "Refrigerator" in str_repr
        assert "150" in str_repr
        assert "1440" in str_repr
    
    def test_appliance_validation_errors(self):
        """Test validation errors."""
        with pytest.raises(ValueError, match="name"):
            Appliance("", 1000.0, 10.0)
        
        with pytest.raises(ValueError, match="power_w"):
            Appliance("Test", -100.0, 10.0)
        
        with pytest.raises(ValueError, match="usage_minutes"):
            Appliance("Test", 1000.0, -5.0)
    
    def test_appliancelist_composition(self):
        """Test ApplianceList composition."""
        app1 = Appliance("TV", 100.0, 120.0)
        app2 = Appliance("Laptop", 50.0, 480.0)
        
        app_list = ApplianceList([app1, app2])
        
        assert len(app_list) == 2
        assert app1 in list(app_list)
        assert app2 in list(app_list)
    
    def test_appliancelist_operator_overloading(self):
        """Test operator overloading."""
        app1 = Appliance("Fan", 50.0, 60.0)
        app2 = Appliance("Light", 20.0, 120.0)
        app3 = Appliance("Phone", 5.0, 60.0)
        
        list1 = ApplianceList([app1, app2])
        list2 = ApplianceList([app3])
        
        combined = list1 + list2
        
        assert len(combined) == 3
        assert app1 in list(combined)
        assert app2 in list(combined)
        assert app3 in list(combined)
    
    def test_appliancelist_total_energy(self):
        """Test total energy calculation."""
        app1 = Appliance("Device1", 1000.0, 30.0)
        app2 = Appliance("Device2", 2000.0, 15.0)
        
        app_list = ApplianceList([app1, app2])
        total = app_list.total_energy_kwh()
        
        assert total == pytest.approx(1.0, rel=1e-3)

