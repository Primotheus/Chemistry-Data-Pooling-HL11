import pandas as pd
import numpy as np
import re

def parse_density(s):
    """
    Converts density strings like "1.6 * 10^3" into numeric values.
    """
    try:
        s = str(s).strip()
        
        # Regex to find patterns like "1.6 * 10^3"
        # It looks for a number (base), an asterisk, "10^3"
        match = re.search(r'([\d\.]+)\s*\*\s*10\^3', s)
        
        if match:
            base = float(match.group(1))
            return base * 1000
        else:
            # If no match, try to convert directly
            return float(s)
    except (ValueError, TypeError, AttributeError):
        # Return NaN if conversion fails
        return np.nan

def analyze_data(file_path):
    """
    Reads the Excel file, calculates, and prints density statistics.
    """
    try:
        df = pd.read_excel(file_path)
        
        # Clean up column names by removing leading/trailing whitespace
        df.columns = df.columns.str.strip()
        
        # Find the density column (robustly)
        density_col_name = None
        for col in df.columns:
            # Check for the column name, even if it has a period at the end
            if 'Density (kg m-3)' in col:
                density_col_name = col
                break
                
        if density_col_name is None:
            print("Error: Could not find the density column.")
            print("Make sure a column with 'Density (kg m-3)' in its name exists.")
            return

        print(f"Analyzing column: '{density_col_name}'")
        
        # Apply the parsing function to the density column
        df['density_numeric'] = df[density_col_name].apply(parse_density)
        
        # Drop any rows where parsing failed
        densities = df['density_numeric'].dropna()
        
        if densities.empty:
            print("Error: No valid density data could be parsed.")
            return

        # --- Calculations ---
        
        # 1. Mean Density
        mean_density = densities.mean()
        
        # 2. Standard Deviation of the dataset
        # ddof=1 for sample standard deviation
        std_dev = densities.std(ddof=1)
        
        # 3. Number of data points
        n = len(densities)
        
        # 4. Standard Error of the Mean (Uncertainty of the Mean)
        sem = std_dev / np.sqrt(n)
        
        # --- Print Results ---
        print("\n--- Density Analysis Results ---")
        print(f"Data file used: {file_path}")
        print(f"Number of valid data points (N): {n}")
        print("\n" + "="*30)
        print(f"Mean Density: {mean_density:.2f} kg/m^3")
        print(f"Standard Deviation: {std_dev:.2f} kg/m^3")
        print("="*30)
        print("\nMean Density with Uncertainty (Mean ± SEM):")
        print(f"({mean_density:.2f} ± {sem:.2f}) kg/m^3")
        print("\nRaw Data Parsed:")
        print(densities.to_string())

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except ImportError:
        print("Error: The 'openpyxl' library is required to read .xlsx files.")
        print("Please install it by running: pip install openpyxl")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    file_path = 'rubberstopper.xlsx'
    analyze_data(file_path)