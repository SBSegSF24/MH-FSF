# MH-FSF: A Framework for Reproduction, Experimentation, and Evaluation of Feature Selection Methods #

In this paper, we present a software framework that integrates the reproduction and implementation of feature selection methods in a modular and integrated manner, named MH-FSF (Malware-Hunter Features Selection Framework). This effort has involved several researchers over the past years. To extensively evaluate the tool, we implemented 17 feature selection methods, grouped into 11 classic and 6 domain-specific methods, using them in the context of Android malware detection.

## [List of implemented and ready-to-use feature selection methods](METHODS.md)

## [How to add new methods in few steps?](NEW_METHOD.md)

## Basic structure

- **main.py**: Evaluates input parameters and executes features selection methods.
- **evaluation.py**: Runs machine learning models, evaluates performance.
- **graphs.py**: Generates and saves various performance visualization graphs.
- **utils.py**: Implements auxiliary functions.
- **methods**: Feature selection methods repository.

## Clonning GitHub repository
```bash

git clone https://github.com/SBSegSF24/MH-FSF.git

cd MH-FSF

```

## Building and running in Docker :whale:

1. Installing Docker and building your image.
```bash

sudo apt install docker docker.io

sudo usermod -aG docker $USER  # Only if you have not executed this command yet. You may need to close and reopen your terminal after issuing it.

docker build -t mhfsf:latest .

```

2. Starting a Docker container in **persistent** or **non persistent** mode.

**Note**: It will take around 3 minutes to run on a Intel Core i7-9700 CPU 3.00GHz, 8 cores, 16 GB RAM.

**Non persistent**: Output files will be deleted when container finishes its execution. 
```bash

docker run -it mhfsf

```
**Persistent**: Output files will be saved and avaliable at the current directory.
```bash

docker run -v $(readlink -f .):/mhfsf -it mhfsf

```

## :penguin: Running in your linux

**Installing Python, if necessary**

~~~sh
sudo apt update
sudo apt install python3
~~~

**Installing default package manager for Python (pip), if necessary**

~~~sh
sudo apt install python3-pip
~~~

**Use Python virtual environment**

~~~sh
sudo apt install python3-venv
python3 -m venv venv
source venv/bin/activate
~~~

**Installing requirements**
~~~sh
pip install -r requirements.txt
~~~

**Running tool** (make sure you are using **python 3.10.12** or higher)
~~~sh
python3 main.py run --all-fs-types --all-fs-methods --all-ml-models -d DATASET [DATASET ...]
~~~

## :dart: Running **demo** scripts

**Option 1**: script will install requirements in your system and run MH - FSF with **some** methods (approximate execution time: 4 min)

```bash

chmod +x run_demo_app_fast.sh

./run_demo_app_fast.sh

```

**Option 2**: script will install requirements in your system and run MH - FSF with **all** methods (approximate execution time: 20 min)

```bash

chmod +x run_demo_app_full.sh

./run_demo_app_full.sh

```

**Option 3**: script will execute will build Docker image and run MH - FSF in fast mode

```bash

chmod +x run_demo_docker.sh

./run_demo_docker.sh

```

## :pushpin: Available arguments:

```
usage: main.py run [-h]
        (--fs-types TYPE [TYPE ...] | --all-fs-types)
        (--fs-methods METHOD [METHOD ...] | --all-fs-methods)
        (--ml-models MODEL [MODEL ...] | --all-ml-models)
        -d DATASET [DATASET ...] [-c CLASS_COLUMN] [-th FLOAT] [--output OUTPUT] [--parallelize]

Arguments:
  -h/--help
            Show help message and exit
  --fs-types TYPE [TYPE ...]
            Types of feature selection (FS) methods
  --all-fs-types
            All types of feature selection (FS) methods
  --fs-methods METHOD [METHOD ...]
            Run selected methods
  --all-fs-methods
            Run all methods
  -d/--datasets DATASET [DATASET ...]
            One or more datasets (csv files).
            For all datasets in directory use: [DIR_PATH]/*.csv
  -c/--class-column CLASS_COLUMN
            Name of class column. Default: class
  --output OUTPUT
            Output file directory. Default: results
  --parallelize
            Parallel execution
```

## :gear: Environments

MH-FSF has been tested in following environments:

**Hardware**: Intel(R) Xeon(R) E5-4617, 64GB RAM. **Software**: Ubuntu 22.04.4 LTS (jammy), Linux version 5.15.0-113-generic (buildd@lcy02-amd64-072), Docker version 24.0.5 (build ced0996), Python 3.10.12

**Hardware**: Apple M2, 8GB RAM. **Software**: Debian GNU/Linux 12 (bookworm), Linux debian-gnu-linux-12 6.1.0-9-arm64, Docker version 20.10.24+dfsg1 (build 297e128), Python 3.11.2

**Hardware**: Intel(R) Core(TM) i7-1185G7, 32GB RAM. **Software**: Ubuntu 22.04.4 LTS (jammy), Linux version 5.15.153.1-microsoft-standard-WSL2, Docker version 24.0.5 (build ced0996), Python 3.10.12

**Hardware**: Intel Core i7-9700 CPU 3.00GHz, 8 cores, 16 GB RAM. **Software**: Debian GNU 11 e 12, Python 3.9.2 e 3.11.2, Docker 20.10.5 e 24.07.


## Link to papers (specific methods)
- FSDroid:
  https://link.springer.com/article/10.1007/s11042-020-10367-w
- SemiDroid:
  https://link.springer.com/article/10.1007/s13042-020-01238-9
- RFG:
  https://www.mdpi.com/2079-9292/9/3/435
- JOWNDroid:
  https://www.sciencedirect.com/science/article/abs/pii/S016740482030359X
- MT:
  https://www.sciencedirect.com/science/article/pii/S1319157821003049
- SigPID:
  https://ieeexplore.ieee.org/document/7888730
- SigAPI:
  https://www.semanticscholar.org/paper/Significant-API-Calls-in-Android-Malware-Detection-Galib-Hossain/a1c5d7097ec467b246ee2c372461934427fac1cd
