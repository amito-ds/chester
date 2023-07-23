import setuptools

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

setuptools.setup(
    name="madcat",
    version="7.3",
    author="Amit Osi",
    author_email="amitosi6666@gmail.com",
    description=open('README.md').read(),
    url="https://github.com/amito-ds/chester",
    packages=setuptools.find_packages(exclude=["tests*"]),
    include_package_data=True,
    python_requires=">=3.8",
    # List of dependencies for this package
    install_requires=install_requires,
    setup_requires=["pytest-runner"]
)