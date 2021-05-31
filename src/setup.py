# Removing the package is also possible with pip uninstall package-name,
# and package name can be read from pip list

from setuptools import setup, find_packages

setup(name='recsys', packages=find_packages())