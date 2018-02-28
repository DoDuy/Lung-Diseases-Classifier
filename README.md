# Diseases Detection from Chest X-ray data
Machine Learning Capstone Project - Udacity MLND

## Project Overview
With so many lung diseases people can get, here is just one example of diseases we can save if we find them out earlier.
With the technology machine and computer power, the earlier identification of diseases, particularly lung disease, we can be helped to detect earlier and more accurately, which can save many many people as well as reduce the pressure on the system. The health system has not developed in time with the development of the population.

## Metrics
F-beta score with β = 0.5 to represent precision will be more important than recall in this case.

## Algorithms and Techniques
• CNN
• Spacial Transformer
• VGG finetuning
• Capsule Network

## Note

1. Following are the file descriptions and URL’s from which the data can be obtained:
```
• data sample/sample_labels.csv: Class labels and patient data for the sample dataset
• data sample/Data_entry_2017.csv: Class labels and patient data for the full dataset
• data sample/images/*: 10 chest X-ray images

Full dataset: https://www.kaggle.com/nih-chest-xrays/data/data
Sample dataset: https://www.kaggle.com/nih-chest-xrays/sample/data

```

2. Following are the notebooks descriptions and python files descriptions, files log:
```
Notebooks:
• Capsule Network - FullDataset.ipynb: Capsule Network with my architecture in full dataset
• Capsule Network - SampleDataset.ipynb: Capsule Network with my architecture in sample dataset
• Capsule Network basic - FullDataset.ipynb: Capsule Network with Hinton's architecture in full dataset
• Capsule Network basic - SampleDataset.ipynb: Capsule Network with Hinton's architecture in sample dataset
• Data analysis - FullDataset.ipynb: Data analysis in full dataset
• Data analysis - SampleDataset.ipynb: data analysis in sample dataset
• Data preprocessing - SampleDataset.ipynb: Data preprocessing
• Demo.ipynb: Demo prediction 20 samples
• optimized CNN - FullDataset.ipynb: My optimized CNN architecture in full dataset
• optimized CNN - SampleDataset.ipynb: My optimized CNN architecture in sample dataset
• vanilla CNN - FullDataset.ipynb: Vanilla CNN in full dataset
• vanilla CNN - SampleDataset.ipynb: Vanilla CNN in sample dataset

Python files
• capsulelayers.py: capsule layer from https://github.com/XifengGuo/CapsNet-Keras
• spatial_transformer.py: spatial transformer layser from https://github.com/hello2all/GTSRB_Keras_STN
So thank you guys for support me with capsule layer and spatial transformer layer in Keras-gpu

Log:
• FullDataset Log: all log file in full dataset
• SampleDataset Log: all log file in sample dataset
```
