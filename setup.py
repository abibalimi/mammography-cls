from setuptools import setup
from setuptools import find_packages


with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]


setup(name="mammography",
      version="1.0",
      description="Lightweight mammography classification package",
      author_email="abib.alimi@gmail.com",
      packages=find_packages(),
      install_requires=requirements,
      )