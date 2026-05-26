from setuptools import find_packages, setup

setup(
    name='m0',
    packages=find_packages(),
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
