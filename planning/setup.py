from distutils.core import setup
from os.path import isdir
from itertools import product


all_packages = ['m0']
packages = list(filter(isdir, all_packages))

setup(
    name='m0',
    packages=packages,
    version='1.1.0',
    install_requires=[
            'matplotlib',
            'numpy',
            'scipy',
            'gymnasium',
            'mujoco',
            'opencv-python',
            'shapely',
            'concave_hull',
            'ompl',
            'stable_baselines3',
            'tensorboard',
            'pynput'],



    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Robotics'
    ]            
            
            
)


    