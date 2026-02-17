#!/usr/bin/env python3
"""
Read the monthly average traffic volumes from a .txt file, predict traffic
counts at each sensor/time using the trained model, and compare predicted
values to the monthly averages.
"""
import argparse
import pandas as pd
import os
from traffic_flow_predictor import TrafficFlowPredictor

def main():
    parser = argparse.ArgumentParser(
        description="Compare model predictions with monthly average volumes."
    )
    parser.add_argument(
        'model_path',
        help='Path to the trained Keras model file'
    )
    parser.add_argument(
        '-o', '--output',
        default='prediction_comparison.txt',
        help='Output text file for comparisons'
    )
    args = parser.parse_args()

    # Load monthly averages
    df_avg = pd.read_csv('datasets/scats_averages.txt')

    # Initialize predictor (loads data and model once)
    predictor = TrafficFlowPredictor('datasets/Scats Data October 2006.xls', args.model_path)

    # Compare predictions against averages
    results = []
    for _, row in df_avg.iterrows():
        sensor = int(row['SCATS_Number'])
        time_str = row['Time']
        avg_vol = float(row['Average_Traffic_Volume'])

        # Predict raw 15-min count for this sensor/time
        pred_vol = predictor.predict(sensor, time_str)

        # Compute difference
        diff = pred_vol - avg_vol
        results.append((sensor, time_str, avg_vol, pred_vol, diff))

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Write comparison to file
    with open(args.output, 'w') as f:
        f.write('SCATS_Number,Time,Average,Predicted,Difference\n')
        for sensor, time_str, avg_vol, pred_vol, diff in results:
            f.write(f"{sensor},{time_str},{avg_vol:.2f},{pred_vol:.2f},{diff:.2f}\n")

    print(f"Comparison written to {args.output}")

if __name__ == '__main__':
    main()
