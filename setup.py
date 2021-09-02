from setuptools import setup, find_packages

setup(
        name='deep_sort_realtime', 
        version='1.0', 
        packages=find_packages(exclude=("test",)),
        package_data={
          'deep_sort_realtime.embedder': ['weights/*']
        },
        install_requires=[
            'numpy',
            'scipy',
        ]
    )
