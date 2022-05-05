# Track-To-Learn

Deep reinforcement learning for tractography.

Update: this work is now published ! 

See preprint: https://www.biorxiv.org/content/10.1101/2020.11.16.385229v1
And published version: https://www.sciencedirect.com/science/article/pii/S1361841521001390

## Installation and setup

It is recommended to use a virtualenv to run the code

``` bash
virtualenv .env
```

Then, run `setup.py` to setup the repository

``` bash
python setup.py develop
```

And then install the dependencies with `pip`

``` bash
pip install -r requirements.txt
pip install -r extra-requirements.txt
```

Finally, set `PYTHONPATH` to include the repository

``` bash
export PYTHONPATH="${PYTHONPATH}:./"
```

## Scripts

In the `scripts` folder, you will find all the scripts used to obtain data for all the experiments in [COMING SOON]. If `DATASET_FOLDER` and `WORK_DATASET_FOLDER` are properly set, running the script as-is should work.

These scripts can be modified to suit your needs if you want to experiment with the framework.

## Data

You can download data from [COMING SOON]. Then, in scripts, you can change `DATASET_FOLDER` to wherever you extracted this.

## Running

Run a script in the `scripts` folder.
To use [Comet.ml](https://www.comet.ml/), follow instructions [here](https://www.comet.ml/docs/python-sdk/advanced/#python-configuration), with the config file either in your home folder or current folder. 

## Contributing

Only Python3.7 is supported for now. Please follow PEP8 conventions, use meaningful commit messages and mark PRs appropriately. Follwing `numpy` standards is heavily recommended: https://numpy.org/doc/1.16/dev/gitwash/development_workflow.html

## Motivations

Rheault et al 2020[1] highlights several limitations, such as the bottleneck effect, wall effectand narrow intersection effect, of classical tractography algorithms. Several machine learning methods for tractography have been proposed[2] but since they are learning from biased data produced by classical tractography algorithms, it is most likely that they will reproduce the same results. 

Deep reinforcement learning is a type of machine learning which does not need ground-truth, or "labelled", data. One could hope that by training a machine learning to do tractography through reinforcement learning, it would not suffer from the same biases as mentionned above. 

All in all, the main assumption is that it is easier to encode anatomical priors through rewards than through cleaned streamlines.

## How to cite

```
@article{theberge2021,
title = {Track-to-Learn: A general framework for tractography with deep reinforcement learning},
journal = {Medical Image Analysis},
pages = {102093},
year = {2021},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2021.102093},
url = {https://www.sciencedirect.com/science/article/pii/S1361841521001390},
author = {Antoine Théberge and Christian Desrosiers and Maxime Descoteaux and Pierre-Marc Jodoin},
keywords = {Tractography, Deep Learning, Reinforcement Learning},
abstract = {Diffusion MRI tractography is currently the only non-invasive tool able to assess the white-matter structural connectivity of a brain. Since its inception, it has been widely documented that tractography is prone to producing erroneous tracks while missing true positive connections. Recently, supervised learning algorithms have been proposed to learn the tracking procedure implicitly from data, without relying on anatomical priors. However, these methods rely on curated streamlines that are very hard to obtain. To remove the need for such data but still leverage the expressiveness of neural networks, we introduce Track-To-Learn: A general framework to pose tractography as a deep reinforcement learning problem. Deep reinforcement learning is a type of machine learning that does not depend on ground-truth data but rather on the concept of “reward”. We implement and train algorithms to maximize returns from a reward function based on the alignment of streamlines with principal directions extracted from diffusion data. We show competitive results on known data and little loss of performance when generalizing to new, unseen data, compared to prior machine learning-based tractography algorithms. To the best of our knowledge, this is the first successful use of deep reinforcement learning for tractography.}
}
```

## References
[1]: Rheault, F., Poulin, P., Valcourt Caron, A., St-Onge, E., & Descoteaux, M. (2020). Common misconceptions, hidden biases and modern challenges of dMRI tractography. Journal of Neural Engineering, 17(1), 11001. https://doi.org/10.1088/1741-2552/ab6aad

[2]: Poulin, P., Jörgens, D., Jodoin, P.-M., & Descoteaux, M. (2019). Tractography and machine learning: Current state and open challenges. Magnetic Resonance Imaging, 64, 37–48. https://doi.org/10.1016/j.mri.2019.04.013

