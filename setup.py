from setuptools import setup, find_packages

setup(
    name="BeatProductionBeast",
    version="0.1.0",
    description="AI-powered beat production assistant",
    author="BeatProductionBeast Team",
    author_email="info@beatproductionbeast.com",
    url="https://github.com/username/BeatProductionBeast",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "torch",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Musicians",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="music, beat production, ai, machine learning, deep learning",
    project_urls={
        "Bug Reports": "https://github.com/username/BeatProductionBeast/issues",
        "Source": "https://github.com/username/BeatProductionBeast",
    },
    long_description="""
    BeatProductionBeast is an AI-powered music production assistant 
    that helps musicians and producers create professional-grade beats 
    using advanced neural networks and audio processing techniques.
    
    It combines various modules for neural beat architecture, 
    audio processing, pattern recognition, and style analysis to 
    generate unique and compelling musical content.
    """,
    long_description_content_type="text/markdown",
    entry_points={
        'console_scripts': [
            'beatbeast=src.cli:main',
            'beatbeast-generate=src.beat_generation.cli:main',
            'beatbeast-analyze=src.style_analysis.cli:main',
        ],
    },
)

