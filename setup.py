from setuptools import setup

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

setup(
    name='TCAP',
    version='0.1',
    packages=['TCAP'],
    url='https://github.com/amito-ds/TCAP',
    author='Amit Osi',
    author_email='amitosi6666@gmail.com',
    install_requires=install_requires,
    long_description=open('README.md').read()
)
