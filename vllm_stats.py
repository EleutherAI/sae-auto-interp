# %%
import re
from collections import defaultdict

def process_log_file(filename):
    totals = defaultdict(int)
    counts = defaultdict(int)
    pattern = r'CompletionUsage\((.+?)\)'

    with open(filename, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                usage_data = match.group(1)
                for item in usage_data.split(', '):
                    key, value = item.split('=')
                    totals[key] += int(value)
                    counts[key] += 1

    averages = {key: totals[key] / counts[key] for key in totals}
    return averages

# Process the log file
log_file = "recall_at_one.log"
averages = process_log_file(log_file)

# Print the results
for category, average in averages.items():
    print(f"Average {category}: {average:.2f}")