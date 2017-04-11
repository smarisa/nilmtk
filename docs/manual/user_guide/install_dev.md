
# Install NILMTK

We recommend using
[Anaconda](https://store.continuum.io/cshop/anaconda/), which bundles
togther most of the required packages. Please download Anaconda for
Python 2.7.x.

After installing Anaconda, please do the following.  We have two
sections below:
[one for Linux or OSX](#install-on-ubuntu-like-linux-variants-debian-based-or-osx)
and another [section for Windows](#install-on-windows).

## Python 3

Python 3 support is experimental and hence please only attempt to use
NILMTK on Python 3 if you are experiences with Python.  On Ubuntu,
please run `sudo apt-get install python3-tk` prior to attempting to
install NILMTK for Python 3.

### Install on Ubuntu like Linux variants (debian based) or OSX

NB: The following procedure is for Ubuntu-like Linux variants (Debian
based). Please adapt accordingly for your OS. We would welcome
installation instructions for other OSes as well.

#### Experimental but probably easiest installation procedure for Unix or OSX

In this section we will describe a fast and simple, but fairly
untested installation procedure.  Please give this a go and tell us in
[the issue queue](https://github.com/nilmtk/nilmtk/issues) if this
process doesn't work for you.  The old Unix and OSX instructions are
further down this page if you need to try them.

Install git, if necessary:

```bash
sudo apt-get install git
```

Download NILMTK:

```bash
cd ~
git clone https://github.com/nilmtk/nilmtk.git
cd nilmtk
```

The next step uses [conda-env](https://github.com/conda/conda-env) to
install an environment for NILMTK, using NILMTK's `environment.yml`
text file to define which packages need to be installed:

```bash
conda env create
source activate nilmtk-env
```

Next we will install
[nilm_metadata](https://github.com/nilmtk/nilm_metadata) (can't yet
install using pip / conda):

```bash
cd ~
git clone https://github.com/nilmtk/nilm_metadata/
cd nilm_metadata; python setup.py develop; cd ..
```

Change back to your nilmtk directory and install NILMTK:

```bash
cd ~/nilmtk
python setup.py develop
```

Run the unit tests:

```bash
nosetests
```

Then, work away on NILMTK :).  When you are done, just do `source
deactivate` to deactivate the nilmtk-env.


#### Old installation procedure for Unix or OSX

- Update anaconda
```bash
conda update --yes conda
```

- Install HDF5 libaries and python-dev
```bash
sudo apt-get install libhdf5-serial-dev python-dev
```

- Install git client
```bash
sudo apt-get install git
```

- Install pip and other dependencies which might be missing from Anaconda
```bash
conda install --yes pip numpy scipy six scikit-learn pandas numexpr
pytables dateutil matplotlib networkx future
```
Note that, if you are using `pip` instead of `conda` then remove
`dateutil` and replace `pytables` with `tables`.

Please also note that there is
[a bug in Pandas 0.17](https://github.com/pydata/pandas/issues/11626)
which causes serious issues with data where the datetime index crosses
a daylight saving boundary.  As such, please do not install Pandas
0.17 for use with NILMTK.  Pandas 0.17.1 was released on the 20th Nov
2015 and includes a fix for this bug.  Please make sure you install
Pandas 0.17.1 or higher.

- Install [NILM Metadata](https://github.com/nilmtk/nilm_metadata).
```bash
git clone https://github.com/nilmtk/nilm_metadata/
cd nilm_metadata
python setup.py develop
cd ..
```

- Install psycopg2
First you need to install Postgres:
```bash
sudo apt-get install postgresql postgresql-contrib
sudo apt-get install postgresql-server-dev-all
pip install psycopg2
```

- Misc. pip installs
```bash
pip install nose coveralls coverage hmmlearn==0.1.1
```

- Finally! Install NILMTK
```bash
git clone https://github.com/nilmtk/nilmtk.git
cd nilmtk
python setup.py develop
cd..
```

- If you wish, you may also run NILMTK tests to make sure the installation has succeeded.
```bash
cd nilmtk
nosetests
```

### Install on Windows

- Install [Anaconda](https://store.continuum.io/cshop/anaconda/), which bundles togther most of the required packages. Please download Anaconda for Python 2.7.x.

- Install [git](http://git-scm.com/download/win) client. You may need to add `git.exe` to your path in order to run nilmtk tests. 

- Install pip and other dependencies which might be missing from Anaconda
```bash
conda install --yes pip numpy scipy six scikit-learn pandas numexpr pytables==3.2.2 dateutil matplotlib networkx future
```

- Install [NILM Metadata](https://github.com/nilmtk/nilm_metadata) from git bash
```bash
git clone https://github.com/nilmtk/nilm_metadata/
cd nilm_metadata
python setup.py develop
cd ..
```

-  Install postgresql support (currently needed for WikiEnergy converter)
Download release for your python environment:
http://www.stickpeople.com/projects/python/win-psycopg/

- [Install VS C++ compiler for Python 2.7](https://www.microsoft.com/en-us/download/details.aspx?id=44266). This is needed for building hmmlearn. 

- Misc. pip installs
```bash
pip install nose pbs coveralls coverage hmmlearn==0.1.1
```

- Finally! Install nilmtk from git bash
```bash
git clone https://github.com/nilmtk/nilmtk.git
cd nilmtk
python setup.py develop
cd..
```

- If you wish, you may also run NILMTK tests to make sure the installation has succeeded
```bash
cd nilmtk
nosetests
```

**It must be noted that nosetests give issues on Windows. However, nilmtk works just fine.**
