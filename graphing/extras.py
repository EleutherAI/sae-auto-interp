
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
    