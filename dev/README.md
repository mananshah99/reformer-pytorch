# cs224n entry points

## `pretrain.py`: pretraining task runner

Arguments:
  - `max_seq_len`: maximum sequence length (default = 512)
  - `n_hashes`: number of hashes (if k-means hashing, number of rerandomizations)
  - `causal`: whether to use an autoregressive model
  - `tied_connections`: whether to use tied connections in reversible layers
  - `kmeans`: whether to use k-means hashing
  - `full_attention`: whether to use full attention

To use with tensorboard on Azure:
```
cd ~/reformer-pytorch/dev/
tensorboard --logdir pretrain_logs_tb --port 6006 --bind_all
```
which can subsequently be accessed locally with `http://40.76.36.52:6006/`. 

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