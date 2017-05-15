# NILMTK for SOR 2017

NILMTK is a toolkit developed by NILM researchers to speed up the pace of
progress in the field. See the
[NILMTK paper](http://arxiv.org/pdf/1404.3878v1.pdf)
and [the original NILMTK repository](https://github.com/nilmtk/nilmtk) and
documentation for more information.

SOR or MS-E2177 Seminar on Case Studies in Operations Research (5 cr) is an
Aalto University course where student teams of three to five learn by working
on a research project given by a company or research organization and report
and compare their achievements with those of their peers.

**This repository is a fork of the NILMTK repository that was used to explore
the toolkit and publicly available NILM datasets. As part of the project,
converters for two synthetic datasets and a dataset provided by Fortum was
developed in order to access the data through NILMTK. Also, a wrapper was
implemented into this NILMTK fork that enables the use of an unsupervised NFHMM
algorithm which was also developed by us as part of the project.** The NFHMM
algorithm was itself developed in R and is available in its [own
repository](https://github.com/smarisa/sor-nilm).

The Fortum dataset is not public. However, the performance of the NFHMM
algorithm was explored also against the two synthetic datasets we generated.
The synthetic datasets as well as the code used to generate them is included in
this fork in the ``test`` directory.

The converters developed for the synthetic datasets and the Fortum dataset are
at ``nilmtk/dataset_converters/sortd/convert_sortd.py`` and
``nilmtk/dataset_converters/fortum/convert_fortum.py``, respectively.

The NFHMM wrapper was implemented to ``nilmtk/disaggregate/nfhmm_wrapper.py``.

NILMTK was also extended with several scripts located at the ``scripts``
folder. The ``fortum_data_converter.py`` was created to prune and reorganize
the data of the Fortum dataset into a more suitable CSV format. The dataset
converter mentioned above was developed to convert the results of this script.
Note that appropriate metadata is required as well as the data which are
currently both unpublished.

The ``dsexp.py`` and ``daexp.py`` scripts were developed in order to explore
datasets and disaggregators, respectively. For the most part, they were means
by which the authors also learned hands-on how to use NILMTK.

The ``comparison.py`` script was used to compare the performance of the NFHMM
algorithm we developed with the CO and FHMM algorithms already implemented in
NILMTK and to generate related material for a report.

Some other files were also created into the repo during the learning and
testing phases like ``nilmtk/disaggregate/dummy.py`` which is an extremely
simple supervised disaggregator developed in order to test and demonstrate how
disaggregators can be developed for NILMTK. Some disaggregation result plots are
also available in the ``results`` folder.

**A report was written to document, present and ponder on the results of the
project but unfortunately it is not public at the moment. Also, while it should
be rather straightforward in principle to test our datasets and NFHMM algorithm
yourself few attempts have been made to make it very easy.** It may be wise to
download, install and familiarize yourself with the main NILMTK repository
first and only then explore and test each extension in this repository.
Familiarity with NILMTK is thus very useful.

Please contact the contributors for more information.

## Contributors

Course project team:

* Samuel Marisa, samuel.marisa, aalto.fi
* Mihail Douhaniaris
* Matias Peljo
* Johan Salmelin

See NILMTK contributors from the
[NILMTK repository](https://github.com/nilmtk/nilmtk).