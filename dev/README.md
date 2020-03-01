# cs224n entry points

## `glue.py`: GLUE task runner

Arguments:
  - `tasks`: comma-delineated list of tasks to train (if empty, all tasks trained sequentially)
  - `gpu`: whether to use GPU or CPU
  - `max_seq_len`: maximum sequence length
  - `recurrence`: whether to use recurrence
  - `lsh_attention`: whether to use LSH attention (a la Reformer) or full QK attention (note, not QKV)
  - `num_train_epochs`: number of epochs to run
  - `num_eval_steps`: number of steps between each evaluation 

To use with tensorboard on Azure:
```
cd ~/reformer-pytorch/dev/
tensorboard --logdir runs --port 6006 --bind_all
```
which can subsequently be accessed locally with `http://40.76.36.52:6006/`. 