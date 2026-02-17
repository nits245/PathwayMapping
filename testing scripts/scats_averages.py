#!/usr/bin/env python3
"""
Calculate the average traffic volume per SCATS sensor and time-of-day over an entire month
and output the results to a text file.
"""
import pandas as pd
from datetime import timedelta, datetime

def main():

    # Load and clean the data
    # Skip the first row (header info) and drop any completely empty rows
    df_raw = (
        pd.read_excel('datasets/Scats Data October 2006.xls', sheet_name='Data', skiprows=1)
        .dropna(how='all')
    )
    df_raw.columns = df_raw.columns.str.strip()

    # Identify the 15-minute interval columns V00 through V95
    interval_cols = [f'V{str(i).zfill(2)}' for i in range(96)]
    required_cols = ['SCATS_Number'] + interval_cols
    df = df_raw[required_cols]

    # Transform from wide to long format
    df_long = df.melt(
        id_vars=['SCATS_Number'],
        value_vars=interval_cols,
        var_name='Interval',
        value_name='Traffic_Volume'
    )

    # Extract interval index and compute time-of-day string
    df_long['Interval_Index'] = (
        df_long['Interval'].str.extract(r'V(\d+)')[0].astype(int)
    )
    df_long['Time'] = df_long['Interval_Index'].apply(
        lambda idx: (datetime.min + timedelta(minutes=15 * idx)).time().strftime('%H:%M')
    )

    # Group by sensor and time, compute the average traffic volume
    avg_df = (
        df_long
        .groupby(['SCATS_Number', 'Time'], as_index=False)
        ['Traffic_Volume']
        .mean()
    )

    # Write results to the output text file
    with open('datasets/scats_averages.txt', 'w') as f:
        # Header
        f.write('SCATS_Number,Time,Average_Traffic_Volume\n')
        # Each line: sensor, time, average volume
        for _, row in avg_df.iterrows():
            sensor = int(row['SCATS_Number'])
            time_str = row['Time']
            avg_vol = row['Traffic_Volume']
            f.write(f"{sensor},{time_str},{avg_vol:.2f}\n")

if __name__ == '__main__':
    main()
