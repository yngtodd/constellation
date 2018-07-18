from setuptools import setup, find_packages


setup(
    name='constellation',
    version="0.0.1",
    packages=find_packages(),
    install_requires=['numpy', 'hyperspace', 'scikit-learn'],

    # metadata for upload to PyPI
    author="Todd Young",
    author_email="youngmt1@ornl.gov",
    description="Ensembling with distributed Bayesian optimization",
    license="MIT",
    keywords="parallel optimization smbo ensemble",
    url="https://github.com/yngtodd/constellation",
)
