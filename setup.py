import setuptools

setuptools.setup(
    name='lobster_common',
    version='0.0.3',
    description='Shared code between Lobster control code and Lobster Simulator',
    url='https://github.com/LOBSTER-Robotics/Common',
    author='Joris Quist',
    author_email='Jorisquist@gmail.com',
    license='',
    packages= setuptools.find_packages(),
    include_package_data=True,
    install_requires=['numpy'],
    classifiers=[],
)
