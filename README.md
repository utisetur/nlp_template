<div align="center">

# PyTorch Lightning template


[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
<!--
ARXIV
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/Thurs88/pl_template/actions/workflows/ci.yml/badge.svg)
[![python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://github.com/Thurs88/pl_template#requirements)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-black)](https://github.com/Thurs88/pl_template/blob/master/.pre-commit-config.yaml)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
 
<!--
Conference
-->
</div>

## Description
Use this template to start new deep learning / ML projects.
Main frameworks used:

* [hydra](https://github.com/facebookresearch/hydra)
* [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)

### Goals
The goal of this project is to create a universal template for training neural networks to solve various problems in the field of genetics, NLP and CV.
In addition this template can be used to structure ML/DL paper-code the same so that results can easily be extended and replicated.

The main ideas of the pipeline:
* all parameter, modules and classes are defined in config files;
* there are configs for different optimizers/schedulers/loggers and so on and you can easy to switch between them;
* using this pipeline, it is easier to share the results with your teammates;

### TO-DO
* make templates for different deep learning tasks
* make project as a package

## How to run
First, install dependencies
```bash
# clone project
git clone https://github.com/Thurs88/pl_template

# install project
cd pl_template
pip install pre-commit
pip install -r requirements.txt
pre-commit install
 ```
Next, navigate to any file and run it.
For example, this command will run training on MNIST (data will be downloaded):
```shell
>>> python train.py --config-name mnist_config model.encoder.params.to_one_channel=True
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
