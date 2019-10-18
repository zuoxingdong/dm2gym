import codecs

from setuptools import setup
from setuptools import find_packages

# Read long description of README markdown, shows in Python Package Index
with codecs.open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Minimal requried dependencies
install_requires = ['numpy',
                    'gym']
tests_require = ['pytest', 
                 'flake8', 
                 'sphinx', 
                 'sphinx_rtd_theme']

setup(name='dm2gym',
      version='0.2.0',
      author='Xingdong Zuo',
      author_email='zuoxingdong@hotmail.com',
      description='dm2gym: Convert DeepMind Control Suite to OpenAI gym environments.',
      long_description=long_description, 
      long_description_content_type='text/markdown',
      url='https://github.com/zuoxingdong/dm2gym',
      install_requires=install_requires,
      tests_require=tests_require,
      python_requires='>=3',
      # List all packages (folder with __init__.py), useful to distribute a release
      packages=find_packages(), 
      # tell pip some metadata (e.g. Python version, OS etc.)
      classifiers=['Programming Language :: Python :: 3', 
                   'License :: OSI Approved :: MIT License', 
                   'Operating System :: OS Independent', 
                   'Natural Language :: English', 
                   'Topic :: Scientific/Engineering :: Artificial Intelligence']
)
