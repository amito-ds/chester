from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

setup(
    name='TCAP',
    version='0.1',
    packages=find_packages(),  # include all packages in all directories
    url='https://github.com/amito-ds/TCAP',
    author='Amit Osi',
    author_email='amitosi6666@gmail.com',
    install_requires=install_requires,
    long_description=open('README.md').read(),
)
