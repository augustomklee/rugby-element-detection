from setuptools import setup, find_packages

setup(
    name="rugby-detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.1",
        "numpy>=1.18.5",
        "opencv-python>=4.1.2",
        "Pillow>=7.1.2",
        "PyYAML>=5.3.1",
        "tqdm>=4.41.0",
        "matplotlib>=3.2.2",
        "seaborn>=0.11.0",
        "roboflow",
    ],
    author="Augusto Min Kyu Lee",
    author_email="augusto.mk.lee@gmail.com",
    description="YOLOv5-based object detection system for rugby analysis",
    keywords="computer-vision, object-detection, yolov5, deep-learning, rugby",
    url="https://github.com/augustomklee/rugby-element-detection",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ],
)