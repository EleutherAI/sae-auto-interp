from collections import defaultdict
import heapq

def find_top_n_files_with_most_both_per_layer(data, n=2):
    both_counts_per_layer = defaultdict(list)
    
    for file, scores in data:
        layer = int(file.split("_")[0].split("layer")[-1])
        
        both_count = sum(1 for score in scores if score['highlighted'] and score['ground_truth'] and score['predicted'])
        
        # Use a min-heap of size n to keep track of top n files
        if len(both_counts_per_layer[layer]) < n:
            heapq.heappush(both_counts_per_layer[layer], (both_count, file))
        elif both_count > both_counts_per_layer[layer][0][0]:
            heapq.heapreplace(both_counts_per_layer[layer], (both_count, file))
    
    # Convert heaps to sorted lists
    top_n_per_layer = {
        layer: sorted([(count, file) for count, file in files], reverse=True)
        for layer, files in both_counts_per_layer.items()
    }
    
    return top_n_per_layer


def tp_fp_fn_tn(df):
    tp = df[(df['highlighted'] == True) & (df['activates'] == True)].shape[0]
    fp = df[(df['highlighted'] == True) & (df['activates'] == False)].shape[0]
    fn = df[(df['highlighted'] == False) & (df['activates'] == True)].shape[0]
    tn = df[(df['highlighted'] == False) & (df['activates'] == False)].shape[0]

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }
    