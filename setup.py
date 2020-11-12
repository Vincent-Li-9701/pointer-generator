from setuptools import setup, find_packages

pkg_name = 'pointer_generator'


setup(
    name=pkg_name,
    version='0.1',
    author='Yanjun Chen, Vincent Li, Angelica Sun',
    description='Stanford CS330 project',
    long_description_content_type='text/markdown',
    packages=[
        package for package in find_packages() if package.startswith(pkg_name)
    ],
)
