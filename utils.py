import os
import torch
import logging
import glob

logger = logging.getLogger(__name__)

def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_dir, max_checkpoints=5):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{iteration}.pth')
    logger.info(f"Saving model and optimizer state at iteration {iteration} to {checkpoint_path}")
    
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    torch.save({
        'model': state_dict,
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'learning_rate': learning_rate
    }, checkpoint_path)

    # Remove old checkpoints
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_')], key=lambda x: int(x.split('_')[1].split('.')[0]))
    if len(checkpoints) > max_checkpoints:
        for ckpt in checkpoints[:-max_checkpoints]:
            os.remove(os.path.join(checkpoint_dir, ckpt))

def load_checkpoint(checkpoint_dir, model, optimizer=None):
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_')], key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
    if not checkpoints:
        return model, optimizer, 0, 0

    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[0])
    assert os.path.isfile(latest_checkpoint)
    checkpoint_dict = torch.load(latest_checkpoint, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        model.module.load_state_dict(saved_state_dict)
    else:
        model.load_state_dict(saved_state_dict)
    
    logger.info(f"Loaded checkpoint '{latest_checkpoint}' (iteration {iteration})")
    return model, optimizer, learning_rate, iteration

def latest_checkpoint_path(dir_path, regex="G_*.pth"):
  f_list = glob.glob(os.path.join(dir_path, regex))
  f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  x = f_list[-1]
  print(x)
  return x