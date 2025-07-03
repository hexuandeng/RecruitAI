import io
import pathlib

import pkg_resources
from setuptools import find_packages, setup

with pathlib.Path('requirement.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

setup(
    name='recruit_ai',
    packages=["recruit_ai"],
    version='0.1',
    description='Recruit AI',
    author='Anonymous',
    install_requires=install_requires,
    entry_points={
        "console_scripts": [],
    },
    package_data={},
    classifiers=["Programming Language :: Python :: 3"],
)
