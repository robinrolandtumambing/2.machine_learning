from setuptools import setup, find_packages

setup(
    name='2.machine_learning',
    version='1.0',
    package=find_packages(where='src'),
    package_dir={'': 'src'}
)

# in the terminal run: pip install -e .