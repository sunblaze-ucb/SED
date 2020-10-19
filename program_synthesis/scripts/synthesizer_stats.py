
import sys

from program_synthesis.analysis.load_results import get_baseline_stats

print(get_baseline_stats(path=sys.argv[1], data_folder='data'))