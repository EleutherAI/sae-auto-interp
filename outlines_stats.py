# %%
import re
from collections import defaultdict

def process_log_file(filename):
    totals = defaultdict(int)
    counts = defaultdict(int)
    pattern = r'(\w+) tokens: (\d+)'

    with open(filename, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                token_type, token_count = match.groups()
                totals[token_type] += int(token_count)
                counts[token_type] += 1

    averages = {key: totals[key] / counts[key] for key in totals}
    return averages

# Process the log file
log_file = "simulation.log"
averages = process_log_file(log_file)

# Print the results
for category, average in averages.items():
    print(f"Average {category} tokens: {average:.2f}")

# Print total counts
total_tokens = sum(averages.values())
print(f"\nTotal average tokens: {total_tokens:.2f}")