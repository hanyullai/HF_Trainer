from pathlib import Path
import shutil
import time
import sys
import os

def del_training_state(checkpoint_dir):
    sorted_dir = [x for x in checkpoint_dir.iterdir() if (x.is_dir() and 'checkpoint-' in x.name)]
    sorted_dir.sort(key=lambda x: int(''.join(c for c in x.name if c.isdigit())))
    for existing_dir in sorted_dir[:-1]: # keep the latest one
        step = int(''.join(c for c in existing_dir.name if c.isdigit()))
        state_dir = existing_dir / f'global_step{step}'
        if existing_dir.is_dir() and state_dir.exists():
            print('deleting training state: ', state_dir)
            shutil.rmtree(state_dir)

if __name__ == '__main__':
    data_path = sys.argv[1]
    if len(sys.argv) == 3:
        mount_pid = sys.argv[2]
    else:
        mount_pid = None
        
    while True:
        if mount_pid is not None and not os.path.exists(f'/proc/{mount_pid}'):
            print('[delete_training_state_process]: mount process is dead, exiting...')
            break
        
        del_training_state(Path(data_path))
        time.sleep(60)