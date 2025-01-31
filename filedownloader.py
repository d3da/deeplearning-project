import os

file_urls = [
    "https://huggingface.co/finnff/GTSRB_trained/resolve/main/final_efficientnet_e10_b64_lr1e-03_20250125_0058.pth?download=true",
    "https://huggingface.co/finnff/GTSRB_trained/resolve/main/final_maxvit_e10_b64_lr1e-03_20250125_0039.pth",
    "https://huggingface.co/finnff/GTSRB_trained/resolve/main/final_regnet_e10_b512_lr1e-03_20250125_0022.pth",
]

for url in file_urls:
    filename = url.split("/")[-1].split("?")[0]
    os.system(f"curl -o {filename} -L {url}")
