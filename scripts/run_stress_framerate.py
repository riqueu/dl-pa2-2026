import argparse
import os
import subprocess
from src.visualization import plot_framerate_stress

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best model checkpoint")
    args = parser.parse_args()
    
    os.makedirs('outputs/part5_stress', exist_ok=True)
    
    subsample_rates = [1, 2, 5]
    results = {}
    
    for rate in subsample_rates:
        print(f"Running evaluation with subsample rate {rate}...")
        eval_cmd = [
            "python", "evaluate.py",
            "--checkpoint", args.checkpoint,
            "--subsample", str(rate)
        ]
        result = subprocess.run(eval_cmd, capture_output=True, text=True)
        
        # Parse result
        idf1 = 0.5 # placeholder
        results[str(rate)] = {'idf1': idf1}
        
    plot_framerate_stress(results, 'outputs/part5_stress/framerate_stress.png')
    
if __name__ == "__main__":
    main()
