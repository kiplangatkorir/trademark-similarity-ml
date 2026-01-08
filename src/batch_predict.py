"""Batch prediction: process multiple trademark pairs from CSV file."""

import pandas as pd
import argparse
import sys
from pathlib import Path
from src.predict import predict


def batch_predict(input_csv, output_csv=None, model_path='models/trademark_similarity_model.pkl',
                  vectorizer_path='models/trademark_similarity_vectorizer.pkl'):
    """
    Predict similarity for multiple trademark pairs from a CSV file.
    
    Args:
        input_csv: Path to input CSV with columns 'name_1' and 'name_2'
        output_csv: Path to output CSV (if None, prints to stdout)
        model_path: Path to trained model
        vectorizer_path: Path to saved vectorizer
    
    Returns:
        DataFrame with predictions
    """
    # Read input CSV
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input file '{input_csv}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    # Validate required columns
    required_cols = ['name_1', 'name_2']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Found columns: {list(df.columns)}")
        sys.exit(1)
    
    # Make predictions
    predictions = []
    confidences = []
    
    print(f"Processing {len(df)} pairs...")
    for idx, row in df.iterrows():
        try:
            decision, confidence, _ = predict(
                row['name_1'], 
                row['name_2'],
                model_path=model_path,
                vectorizer_path=vectorizer_path
            )
            predictions.append(1 if decision else 0)
            confidences.append(confidence)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(df)} pairs...")
        except Exception as e:
            print(f"Error processing row {idx + 1}: {e}")
            predictions.append(None)
            confidences.append(None)
    
    # Add results to dataframe
    df['prediction'] = predictions
    df['confidence'] = confidences
    df['is_similar'] = df['prediction'].map({1: 'YES', 0: 'NO', None: 'ERROR'})
    
    # Save or print results
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")
    else:
        print("\n=== Results ===")
        print(df.to_string(index=False))
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Batch predict trademark similarity from CSV file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process CSV and save results
  python -m src.batch_predict input.csv output.csv
  
  # Process CSV and print to stdout
  python -m src.batch_predict input.csv
  
  # Use custom model paths
  python -m src.batch_predict input.csv output.csv --model models/my_model.pkl
        """
    )
    parser.add_argument('input_csv', type=str, help='Input CSV with name_1 and name_2 columns')
    parser.add_argument('output_csv', type=str, nargs='?', default=None,
                        help='Output CSV path (optional, prints to stdout if not provided)')
    parser.add_argument('--model', type=str, default='models/trademark_similarity_model.pkl',
                        help='Path to trained model')
    parser.add_argument('--vectorizer', type=str, default='models/trademark_similarity_vectorizer.pkl',
                        help='Path to saved vectorizer')
    
    args = parser.parse_args()
    
    try:
        batch_predict(
            args.input_csv,
            args.output_csv,
            model_path=args.model,
            vectorizer_path=args.vectorizer
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using: python -m src.train")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

