# Independent-Component-Analysis

ICA is used for seperating a multivariate signal into additive subcomponents by primary assumption of sources are non gaussian and statistically independent.
It is a special case of BSS (Blind Source Separation) and the goal is to decompose a multivariate signal into its independent non gaussian.

### What's In?
- ica.py : ICA implementation using FastICA Sklearn library
- FastICA.ipynb : Jupyter notebook
- FastICAS.ipynb : Simple code without audio and given as example in Sklearn
- Input1.zip and Input2.zip : Contains the inputs and results obtained
- MusicDominated.wav, VoiceDominated.wav are the observed signals and voice_comp.wav, music_comp.wav are the extracted components used in presentation
- Presentation.key: PPT

### Usage
- python ica.py <observed1.wav> <observed2.wav>

### Result
There will be a folder created in current directory "ICA Components", which contains the individual sources and plots for reference.

### Note
The code accepts only two components means two observations. If we need to extract more components, have to modify slightly.
