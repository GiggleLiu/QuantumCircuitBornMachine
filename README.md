# Quantum Circuit Born Machine - the Demo
Gradient based training of Quantum Circuit Born Machine (QCBM)

## Table of Contents
This project contains

* `notebooks/qcbm_gaussian.ipynb` (or [online](https://drive.google.com/file/d/1LfvWuM8rUPOtdWFRbUhSyjn35ndR7OW6/view?usp=sharing)), basic tutorial of training 6 bit Gaussian distribution using QCBM,
* `notebooks/qcbm_advanced.ipynb` (or [online](https://drive.google.com/file/d/1cA5niJga7aLcJqIdBtpGV9i0vyOen1Aq/view?usp=sharing)), an advanced tutorial,
* `qcbm` folder, a simple python project for productivity purpose.

## Preparations
You may use **either** local or online accesses to our python notebooks.

### Local
Set up your python environment

* python 3.6
* install python libraries

If you want to read notebooks only and do not want to use features like [`projectq`](https://github.com/ProjectQ-Framework/ProjectQ), having `numpy`, `scipy` and `matplotlib` is enough.
To access advanced features, you should install `fire`, `projectq` and `climin`.
```bash
$ conda install pybind11
$ pip install -r requirements.txt
```

Clone this repository [https://github.com/GiggleLiu/gbm.git](https://github.com/GiggleLiu/gbm.git) to your local host.

### Online
1. Sign up and sign in [Google drive](https://drive.google.com/)
2. Connect Google drive with [Google Colaboratory](https://colab.research.google.com)
    - right click on google drive page
    - More
    - Connect more apps
    - search "Colaboratory" and "CONNECT"
3. You can make a copy of notebook to your google drive (File Menu) to save your edits.

## Run Bar-and-Stripes Demo at Your Localhost

```bash
$ ./program.py checkgrad  # check the correctness of gradient
$ ./program.py statgrad  # check gradient will not vanish as layer index increase.
$ ./program.py vcircuit  # visualize circuit using ProjectQ
$ ./program.py train   # train and save data.
$ ./program.py vpdf   # see bar stripe dataset PDF
$ ./program.py generate  # generate bar and stripes using trainned circuit.
```

## Documentations

* paper: *to be added*
* slides: *to be added*

## Authors

* Jin-Guo Liu <cacate0129@iphy.ac.cn>
* Lei Wang <wanglei@iphy.ac.cn>
