import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="grocerypathsim",
    version="0.0.1",
    author="Kimberly Holmgren",
    author_email="kimberly.m.holmgren@gmail.com",
    description="Simulation to estimate customer walking path given proposed "
                "block layout and department assignments.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        'matplotlib>=3.1.0',
        'numpy>=1.18.0',
        'opencv-python',
        'pandas>-1.0.3',
        'pathfinding==0.0.4',
        'pytesseract==0.3.6',
        'scikit-learn==0.22.1',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)