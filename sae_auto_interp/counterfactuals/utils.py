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
