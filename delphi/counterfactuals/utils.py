import subprocess
import torch

def garbage_collect():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_git_info():
    try:
        commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        remote_url = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url']).decode('ascii').strip()
        
        if remote_url.startswith('git@github.com:'):
            remote_url = 'https://github.com/' + remote_url.split('git@github.com:')[1]
        
        if remote_url.endswith('.git'):
            remote_url = remote_url[:-4]

        return f"{remote_url}/tree/{commit_id}"
    except subprocess.CalledProcessError:
        return None  # Not a git repository or git not installed


LAYER_TO_L0 = {   
    0:30,
    1:33,
    2:36,
    3:46,
    4:51,
    5:51,
    6:66,
    7:38,
    8:41,
    9:42,
    10:47,
    11:49,
    12:52,
    13:30,
    14:56,
    15:55,
    16:35,
    17:35,
    18:34,
    19:32, 
    20:34,
    21:33,
    22:32,
    23:32,
    24:55,
    25:54,
    26:32,
    27:33,
    28:32,
    29:33,
    30:32,
    31:52,
    32: 51,
    33:51,
    34:51,
    35:51,
    36: 51,
    37:53,
    38:53,
    39:54,
    40: 49,
    41:45,
}