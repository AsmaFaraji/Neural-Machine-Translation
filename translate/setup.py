from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
  'tensor2tensor'
]

setup(
    name='translate',
    version='0.1',
    author = 'Asma',
    author_email = 'farajiasma@gmail.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='translation',
    requires=[]
)
