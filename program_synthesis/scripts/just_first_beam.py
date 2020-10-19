
import os

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--input-file", required=True)
parser.add_argument("--output-file", required=True)
parser.add_argument("--start", type=float, required=True)
parser.add_argument("--end", type=float, required=True)

args = parser.parse_args()

assert not os.path.exists(args.output_file)

with open(args.input_file) as f:
    data = json.load(f)

for datum in data:
    datum['beams'][1:] = []
    datum['beams_correct'][1:] = []
    
with open(args.output_file, "w") as f:
    json.dump(data, f)