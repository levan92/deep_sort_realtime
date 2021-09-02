from setuptools import setup, find_packages

setup(name='deep_sort_realtime', 
        version='1.0', 
        packages=['deep_sort_realtime'],
        package_data={
          'embedder': ['mobilenetv2/mobilenetv2_bottleneck_wts.pt', 'mobilenetv2_tf/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5']
        },
        install_requires=[
            'numpy',
            'scipy',
        ]
    )
