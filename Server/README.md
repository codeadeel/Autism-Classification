# Server Scripts

* [**Introduction**](#introduction)
* [**Autism Classification Inference Server**](#sc_infer_server)
* [**Autism Classification Inference Client**](#sc_infer_client)

## <a name="introduction">Introduction

Server scripts are used to power server client architecture. Server scripts are responsible for inference handling and client identification. Server architecture makes it possible to load trained resources automatically for inference, and listens for client for request. Server client architecture communication is based on REST API.

## <a name="sc_infer_server">Autism Classification Inference Server

Autism classification inference server is responsible of batched inference, results computation and handling of clients. This [script][ins] take following arguments as input:

```bash
usage: inference_server.py [-h] [-r RESOURCES] [-thres SIM_THRES]
                           [-btch BATCH_SIZE] [-data BASE_DATA]

Autism Classification Inference Server.

optional arguments:
  -h,     --help            show this help message and exit
  -r,     --resources       Absolute Address of Trained Resources Directory
  -thres, --sim_thres       Similarity Threshold in Case of Clustering Mode
  -btch,  --batch_size      Batch Size to Create Base Cluster Embeddings
  -data,  --base_data       Absolute Address of Base Clustering Data Directory
```

## <a name="sc_infer_client">Autism Classification Inference Client

Client scripts are used to request images for inference to server. In simple words, they send batch of images to inference server, and return results after process. Usage for this [script][inc] is given as following:

```python
from inference_client import *

autism_classification_inference_server_ip = 'http://172.17.0.2:8080'

img1 = 'some image path'
img2 = 'some image path'

img_batch = [img1, img2, ...]

target_server = Classify(autism_classification_inference_server_ip)

results, status_code = target_server(img_batch)

```

[ins]: ./inference_server.py
[inc]: ./inference_client.py