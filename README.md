# Anomaly Detection for SWaT Dataset using Sequence-to-Sequence Neural Networks

This repo includes source codes and pre-trained models for the paper, "Anomaly Detection for Industrial Control Systems Using Sequence-to-Sequence Neural Networks." (The paper will be public soon.)

## Requirements
- Python 3
- PyTorch
- InfluxDB (not mandatory, the codes give the output to InfluxDB)

## Workflow

1. parser.py for normalization
2. train.py for training models
3. validate.py for calculating distances
4. judge.py for rating suspicious region (S, described in the paper section 2.5)
5. extract_anomaly_probs.py for extracting S to a file
6. measure_extractor.py for the result (dat/detection)

## Folders and files

### checkpoints/

This folder has pre-trained models.

Filename format: SWaT-P{A}-{B}.net

A means [Process ID].
SWaT has six processes.
We have six models for each process.

B means [Duplicate training ID].
We made two independent training sessions for each model (process), and chose a better one, which has smaller training loss.

### dat/

This folder has two files.

1) SWaT.stat : It includes the list of min and max of all tags. The list is extracted from the normal dataset. All datasets (normal and attack) are going to be normalized with this list.

2) detection : It includes the detection results (positives). Every line has 6 numbers (0 or 1), comma-separated, for process ID sequentially. 1 means positive; 0 means negative. For example, '1,1,0,0,0,0' means the model detected an attack on process #1 and #2.

### evaluation/

This folder includes custom-evaluation codes.
We made the range-based false positive/negative checker.
We published the details to another paper.

In short, as mentioned in the paper, we regard any detection in 'the attack range + 15 minutes' as a true positive, while the precision-and-recall score comes from sample-by-sample answers (labels).
The reason we added 15 minutes is why industrial control systems need some time to get stable after the attack.

Evaluation.py reads the result file (dat/detection) and prints output.
You have to specify the input file with -i.
> ex) python3 Evaluation.py -i ../dat/detection

parameters.cfg includes configurations.
'extend_tail=900' means 900 seconds (15 minutes).

### conf.py

Information about tags, sliding windows, RNN layers, and RNN cell sizes.

### config.ini

Hyper-parameters.

[train] section is for the number of epochs.

[judge] section is for anomaly decision (section 2.5 in the paper).

### db.py

You can use any database you like.
I chose [InfluxDB](https://www.influxdata.com/products/influxdb-overview/) for saving all results.
Most of the figures (graphs) are the screenshot of [Grafana](https://grafana.com/) pages.

### extract_anomaly_probs.py

This script extracts all rates for suspicious region (S).

### first_look.py

This script prints the list of min, max, mean, and std-dev of all tags, all dataset.

### judge.py

**judge.py must run after validate.py ran.**

This script calculates all rates for suspicious region (S) from distances in DB.

### measure_extractor.py

This script makes 'detection' file above (dat/detection) from the result of extract_anomaly_probs.py.

### model.py

The seq2seq model is described in this file.

### network.py

network.py is a helper script for defining model and optimizer, training, inferencing, and saving/loading the model.

### parser.py

This script normalizes the dataset.

### swat_dataset.py

This script defines the PyTorch-style dataset for our pre-processed (normalized) one (from parser.py).

### train.py

This script is for training the model.
According to your environment, you can change GPU-related, DB-related codes.

### validate.py

This script is for measuring distances.
