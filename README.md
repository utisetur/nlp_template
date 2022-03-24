<div align="center">

# PyTorch Lightning NLP template

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.5+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.1-89b8cd?style=for-the-badge&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

A clean and scalable template to kickstart your deep learning project <br>
Click on [<kbd>Use this template</kbd>](https://github.com/utisetur/nlp_template/generate) to initialize new repository.

</div>

<br><br>

## Description
Use this template to start new deep learning / ML projects.
Main frameworks used:

* [hydra](https://github.com/facebookresearch/hydra)
* [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)

### Goals

The goal of this project is to create a universal template for training neural networks to solve various NLP problems. <br>
In addition this template can be used to structure ML/DL paper-code the same so that results can easily be extended and replicated. <br>

### The main ideas of the pipeline
* all parameter, modules and classes are defined in config files;
* there are configs for different optimizers/schedulers/loggers and so on and you can easy to switch between them;
* using this pipeline, it is easier to share the results with your teammates;

## Project Structure

The directory structure of new project looks like this:

```
├── cfg                   <- Hydra configuration files
│   ├── augmentations            <- custom augmentations
│   ├── callbacks                <- Callbacks parameters
│   ├── datamodule               <- Datamodule parameters
│   ├── inference                <- inference/test parameters
│   ├── logging                  <- logger parameters
│   ├── loss                     <- custom losses
│   ├── metrics                  <- custom metrics configs
│   ├── model                    <- model configs
│   ├── optimizer                <- optimizer configs
│   ├── private                  <- logger api and other private info
│   ├── scheduler                <- lr schedulers
│   ├── trainer                  <- Trainer parameters
│   ├── training                 <- training parameters (lr, metrics, freeze backbone, etc)

│   │
│   └── rte_config.yaml       <- Main config for training (RTE task for example)
│
├── data                   <- project data storage
│
├── notebooks              <- Jupyter notebooks
│
├── outputs                <- training results (saved models, test predictions, etc)
│
├── pipeline               <- pipeline source code
│   ├── datamodules              <- Lightning datamodules
│   ├── datasets                 <- Lightning/custom datasets
│   ├── losses                   <- custom losses
│   ├── metrics                  <- custom metrics
│   ├── models                   <- PyTorch/Lightning models
│   ├── schedulers               <- custom lr schedulers
│   ├── trainers                 <- Lightning wodel wrappers/trainers
│
├── src                    <- auxiliary functions
│   ├── ml_utils                 <- model utils
│   ├── technical_utils          <- other utils
│
├── tests
│   ├── ...
│   ├── ...
│   └── ...
│
├── .gitignore                <- List of files/folders ignored by git
├── .pre-commit-config.yaml   <- Configuration of pre-commit hooks for code formatting
├── requirements.txt          <- File for installing python dependencies
└── README.md
```
<br>


## How to run
First, install dependencies
```bash
# clone project
git clone https://github.com/utisetur/nlp_template

# install project
cd nlp_template
pip install pre-commit
pip install -r requirements.txt
pre-commit install
 ```
Next, navigate to any file and run it.
For example, this command will run training on MNIST (data will be downloaded):
```shell
>>> python run_experiment.py --config-name='rte_config'
```

### Citation
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
