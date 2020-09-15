# Tensorizing recurrent neural network architectures for model compression

[Paper]()

### Repository structure
* ``tensorized_rnn`` contains our implementations of GRU and LSTM as well as their tensorized counterparts, 
TT-GRU and TT-LSTM. 
* ``t3nsor`` contains the tensor library used (see README within). 
* ``experiments`` contains models and experiments performed using our tensorized architectures.
 
### System requirements
* Python 3.6
* CPU or NVIDIA GPU + CUDA

### Dependencies and versions
* ``pytorch 1.1.0``
* ``torchvision 0.3.0``
* ``numpy 1.15.4``
* ``sympy 1.5.1``
* ``scipy 1.1.0``
* ``matplotlib 3.1.3``

### Additional dependencies for speaker verification
* ``webrtcvad 2.0.10``
* ``librosa 0.6.2``
* ``umap-learn 0.4.2``
* ``tqdm  4.43.0``
* ``multiprocess 0.70.9``
* ``comet-ml 3.1.6``

### Running experiments

##### Sequential (Permuted) MNIST
```
python pmnist_test.py --epochs 5 --permute --tt --ncores 2 --ttrank 4

```
##### Speaker Verification

Download the [LibriSpeech dataset](http://www.openslr.org/12). Then preprocess
```
python encoder_preprocess.py -r /path/to/raw/dataset/root -o /path/to/output/dir -d librispeech_other
```

Train
```
python encoder_train.py --clean_data_root /path/to/output/dir -m /dir/to/save/models/ -v 50 -u 100
```

### Authors
* Charles C Onu
* Jacob Miller
* Doina Precup

If you use this code in your research, please cite our work:
```
Coming soon...
```