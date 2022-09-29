# Model Training

* [**Introduction**](#introduction)
* [**Trainer**](#trainer)

## <a name="introduction">Introduction

The subject training script is used to process data and subject deep learning model. After training process is completed, resources are saved into some persistent storage.

## <a name="trainer">Trainer

Training on the subject data can be automatically done and handled by training pipeline. This [script][trainer] takes following arguments as input:

```bash
usage: trainer.py [-h] -e EPOCHS [-l LR] [-bs BATCH_SIZE] [-ts TRAIN_SHUFFLE]
                  [-sim SIM_THRES] [-tm] [-sd SEED] [-d DATA] [-tra TRADDR]

Autism Classification Model Trainer.

optional arguments:
  -h,   --help            show this help message and exit
  -e,   --epochs          Number of Epochs to Which Model Should be Trained Upto
  -l,   --lr              Learning Rate for Model Training
  -bs,  --batch_size      Batch Size of Input Data
  -ts,  --train_shuffle   Boolean Valiable to Shuffle Training Data ( True / False )
  -sim, --sim_thres       Similarity Threshold for Metrics Computation
  -tm,  --train_metric    Flag to Compute Extensive Training Metrics
  -sd,  --seed            Seed Value to Randomize Dataset
  -d,   --data  Absolute  Aaddress of the Parent Directory of Data Distribution Sub-Directories
  -tra, --traddr          Absolute Address to Save Training Output
```

[trainer]: ./trainer.py