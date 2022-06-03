# zzsn-nyse
NYSE modeling with R-Transformer and GRU.  
Data source: https://www.kaggle.com/datasets/dgawlik/nyse

## Setup environment
```shell
$ conda env create -f environment.yml
$ conda activate zzsn-nyse
```

## Run experiments
```shell
$ python main.py
```

### Execution options
```shell
$ python main.py --option=value
```
For example:
```shell
$ python main.py --batch_size=32 --seed=555
```
* --features   (type=str, default=chl)
  * features to train the model on, any combination of the following values are possible:
    * c (close price)
    * o (open price)
    * h (high price)
    * l (low price)
* --targets   (type=str, default=c)
  * targets to be predicted by the model, options same as above
* --dropout   (type=float, default=0.15)
  * dropout rate
* --clip   (type=float, default=0.15)
  * gradient clipping value
* --epochs   (type=int, default=30)
  * number of epochs
* --ksize   (type=int, default=7)
  * R-Transformer's number of keys
* --n_level   (type=int, default=3)
  * R-Transformer's number of blocks
* --log_interval   (type=int, default=500)
  * epoch number and loss are displayed after every log_interval
* --lr   (type=float, default=0.1)
  * learning rate
* --model   (type=str, default=RT)
  * model to be trained - either GRU or RT (R-Transformer)
* --rnn_type   (type=str, default=GRU)
  * RNN used in R-Transformer - options are: GRU, LSTM, RNN
* --d_model   (type=int, default=32)
  * R-Transformer's keys and values dimension
* --n   (type=int, default=1)
  * if model=GRU - number of GRU layers
  * if model=RT - number of RNN layers
* --h   (type=int, default=6)
  * if model=GRU - size of hidden state
  * if model=RT - size of attention's hidden state
* --batch_size   (type=int, default=64)
  * batch size
* --proj_len   (type=int, default=1)
  * projection length - how many days to predict
* --seq_len   (type=int, default=25)
  * sequence length - length of input length
* --seed   (type=int, default=1111)