#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

__author__ = 'Matthew Wiesner, Kiran Karra'
__email__ = 'wiesner@jhu.edu, kiran.karra@jhu.edu'
__version__ = '0.0.1'

install_requires = [
    'networkx',
    'lhotse',
]

setuptools.setup(
    name='nachos',
    version=__version__,

    description='Nearly Automatic Creation of Held-out Splits',
    long_description=long_description,
    long_description_content_type="text/markdown",

    url = 'https://github.com/m-wiesner/nachos',

    author=__author__,
    author_email=__email__,

    license='Apache License 2.0',

    python_requires='>=3.8',
    packages=setuptools.find_packages(),

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='design of experiments, speech recognition translation deep-learning multi-modal',

    # Needed for albumentations install
    dependency_links=[],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html

    install_requires=install_requires,

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
          'test': []
    },

    # add installable scripts here
    scripts=[],

    zip_safe=False
)
