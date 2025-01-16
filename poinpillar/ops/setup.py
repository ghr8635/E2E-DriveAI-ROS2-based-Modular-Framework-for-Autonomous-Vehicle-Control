from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
from shutil import copyfile

class CustomBuildExtension(BuildExtension):
    """Custom build extension to ensure .so files are copied to the correct location."""
    
    def run(self):
        # Call the standard build process
        super().run()
        
        # Define the source and destination of .so files
        build_lib = self.build_lib
        so_files = ['voxel_op', 'iou3d_op']
        
        # Copy the built .so files to the package directory
        for so_file in so_files:
            so_filename = f'{so_file}.cpython-310-x86_64-linux-gnu.so'
            source_path = os.path.join(build_lib, 'ops', so_filename)
            destination_path = os.path.join('ops', so_filename)
            if os.path.exists(source_path):
                print(f"Copying {so_filename} to {destination_path}")
                copyfile(source_path, destination_path)
            else:
                print(f"Warning: {so_filename} not found at {source_path}")

setup(
    name='ops',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='ops.voxel_op',
            sources=[
                'voxelization/voxelization.cpp',
                'voxelization/voxelization_cpu.cpp',
                'voxelization/voxelization_cuda.cu',
            ],
            define_macros=[('WITH_CUDA', None)]
        ),
        CUDAExtension(
            name='ops.iou3d_op',
            sources=[
                'iou3d/iou3d.cpp',
                'iou3d/iou3d_kernel.cu',
            ],
            define_macros=[('WITH_CUDA', None)]
        )
    ],
    cmdclass={
        'build_ext': CustomBuildExtension
    },
    include_package_data=True,
    package_data={
        'ops': ['*.so']  # Ensure .so files are included during installation
    },
    zip_safe=False
)
