

UDL Project : GTSRB - German Traffic Sign Recognition Benchmark Adverserial Model

Models and Training Data can be downloaded from:
https://huggingface.co/finnff/GTSRB_trained


* final_efficientnet_e10_b16_lr1e-03_20250107_0335.pth 637 MB
* final_maxvit_e10_b16_lr1e-03_20250107_0527.pth 366 MB
* final_regnet_e10_b128_lr1e-03_20250107_0233.pth 66.8 MB
* data.tar.gz 11 GB


## Filestructure

```
.
├── data #(run `tar -xzf data.tar.gz` after downloading data.tar.gz)
├── data.tar.gz  # (downloadable from huggingface)
├── docker-compose.yml
├── Dockerfile
├── models #(retraining models will end up here if running docker compose up)
├── requirements.txt
└── src
```

## Usage

### Retraining the Base models on the GTSRB dataset

1. Download the data.tar.gz from huggingface and extract using `tar -xzf data.tar.gz`
2. (optional) adjust docker-compose.retrain.yml's shm_size from 8gb if needed
3. (optional) adjust src/train.py's batch size, learning rate, epochs, etc. ~line 225
3. Run `docker compose -f docker-compose.retrain.yml up` to retrain the models on the GTSRB dataset


### Running the Adverserial Model

1. take final_regnet_e10_b128_lr1e-03_20250107_0233.pth and place it in the data dir
2. `docker compose up` 

