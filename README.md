

Test and Validation Set Images for each model (both with and without Defence Transform) can be found in Image Samples [https://github.com/d3da/deeplearning-project/tree/main/results](https://github.com/d3da/deeplearning-project/tree/main/results) 


GTSRB Fine-Tuned Models (and Training Data) can be found on [https://huggingface.co/finnff/GTSRB_trained/tree/main](https://huggingface.co/finnff/GTSRB_trained/tree/main)

![image](https://github.com/user-attachments/assets/e59af0fc-05b9-48c7-90f1-8a9d28a763b9)


UDL Project : GTSRB - German Traffic Sign Recognition Benchmark Adverserial Model

Our Ensemble Models Requires a large (16+ GB) amount of VRAM to work with our given batchsizes (and may not even run at all on very low amounts of VRAM). The code shown here has been ran on RTX 3090 and 4090s nodes from [https://cloud.vast.ai/](https://cloud.vast.ai/)


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

