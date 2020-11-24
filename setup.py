from setuptools import setup, find_packages

setup(
    name='MIP_oracle2',
    packages=find_packages(include=['MIP_oracle2', 'MIP_oracle2.*']),
    install_requires=[
        'Pyomo',
    ]
)
