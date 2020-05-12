from distutils.core import setup

# http://www.diveintopython3.net/packaging.html
# https://pypi.python.org/pypi?:action=list_classifiers

with open('README.txt') as file:
    long_description = file.read()

setup(
    name='sxpid',
    packages=['sxpid'],
    version='1.0',
    description='Shared Exclusion Partial Information Decomposition',
    author='Abdullah Makkeh',
    author_email='abdullah.makkeh@gmail.com',
    url='https://github.com/abzinger/SxPID',
    long_description=long_description,
    classifiers=[
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Development Status :: 5 - Production/Stable",
    "Operating System :: POSIX :: Linux",
    "Intended Audience :: Science/Research",
    "Environment :: Console",
    "Environment :: Other Environment",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: Information Analysis",
    ]
)
