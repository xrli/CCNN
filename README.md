# Concat Convolutional Neural Network for Pulsar Candidate Selection

This repo contains the code and trained models for out paper Concat Convolutional Neural Network for Pulsar Candidate Selection.

### Requirements

- Presto(<https://github.com/scottransom/presto>)

- numpy

- scipy

- pandas

- scikit-learn(>=0.15)

- tensorflow(>=1.8.0)

- keras

- keras_metrics

  Note: The Python version depends on the what version of python is Presto installed on. In other words, the code of CCNN can run both in Python 2 and 3.

### Experimental data

- Download website: <https://github.com/dzuwhf/FAST_label_data

### Usage

- Training a new model:

  ```shell
  python training.py
  ```

- Test:

  ```shell
  python predict.py
  ```

### Citation

- If you found this code useful please cite our paper

