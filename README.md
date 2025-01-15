

UDL Project : GTSRB - German Traffic Sign Recognition Benchmark Adverserial Model

Models and Training Data can be downloaded from:
https://huggingface.co/finnff/GTSRB_trained


* final_efficientnet_e10_b16_lr1e-03_20250107_0335.pth 637 MB
* final_maxvit_e10_b16_lr1e-03_20250107_0527.pth 366 MB
* final_regnet_e10_b128_lr1e-03_20250107_0233.pth 66.8 MB
* data.tar.gz 11 GB


## Filestructure

.
├── data #(run `tar -xzf data.tar.gz` after downloading data.tar.gz)
├── data.tar.gz  # (downloadable from huggingface)
├── docker-compose.yml
├── Dockerfile
├── models #(retraining models will end up here if running docker compose up)
├── requirements.txt
└── src

