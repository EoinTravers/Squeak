from setuptools import setup
# from distutils.core import setup

setup(
    name='Squeak',
    version='0.2.0',
    url='https://github.com/EoinTravers/Squeak',
    author='Eoin Travers',
    author_email='etravers01@qub.ac.uk',
    description=('Analyse mouse trajectory data.'),
    license='GPL',
    packages=['squeak'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Operating System :: Unix',
        'Programming Language :: Python :: 2',
    ],
    install_requires=[
        'numpy >= 1.7.1',
        'pandas >= 0.12.0',
        'scipy >= 0.12.0',
        'matplotlib >= 1.3.0'
    ]
)
