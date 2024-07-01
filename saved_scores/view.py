# %%

import json
import os
from sklearn.metrics import confusion_matrix

def load_files(directory):
    data = []
    for filename in os.listdir(directory):
        if "1547" in filename:
            continue
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as file:
                data.extend(json.load(file))
    return data

def parse_data(data):
    parsed_data = {
        'y_true': [],
        'y_marked': []
    }
    for entry in data:
        parsed_data['y_true'].append(int(entry['is_correct']))
        parsed_data['y_marked'].append(int(entry['marked']))
    return parsed_data

def calculate_tpr_fpr(y_true, y_marked):
    tn, fp, fn, tp = confusion_matrix(y_true, y_marked).ravel()
    tpr = tp / (tp + fn)  # True Positive Rate
    fpr = fp / (fp + tn)  # False Positive Rate
    return tpr, fpr

# %%

directory = '/share/u/caden/sae-auto-interp/saved_scores'  # Replace with the path to your directory
data = load_files(directory)
parsed_data = parse_data(data)

# Calculate TPR and FPR
tpr, fpr = calculate_tpr_fpr(parsed_data['y_true'], parsed_data['y_marked'])

print(f"True Positive Rate (TPR): {tpr}")
print(f"False Positive Rate (FPR): {fpr}")
# %%


def plot_curves(y_true, y_scores):
    # Compute Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    prc_auc = auc(recall, precision)
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot Precision-Recall curve
    plt.figure()
    plt.plot(recall, precision, label=f'PRC curve (area = {prc_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.show()

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()

def main():
    directory = '/share/u/caden/sae-auto-interp/saved_scores'  # Replace with the path to your directory
    data = load_files(directory)
    parsed_data = parse_data(data)
    print(parsed_data)
    plot_curves(parsed_data['y_true'], parsed_data['y_scores'])

if __name__ == "__main__":
    main()