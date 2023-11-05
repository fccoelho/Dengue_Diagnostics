# Dengue_Diagnostics
This is a Gymnasium environment for learning optimal policies for acurately detecting dengue cases in the presence of 
other arbovirus cases such as chikungunya.

## About this environment
In the `DengueDiag-v0` environment, there is two epidemics going on on the same city, dengue and chikungunya. The agent
has to decide, for every case reported, whether to test for dengue or chikungunya, or not to test at all, accepting 
accepting the clinical diagnosis of the Doctor that reported the case. 
The agent receives a reward of -1 for every test performed, and a reward of episize (total number of cases reported) - the
number of cases misdiagnosed. The agent receives a reward of -10 (?) for every case that is not tested and is misdiagnosed.

### Observation Space
The observation space is comprised of 4 arrays:
```python
{
    "clinical_diagnostic": obs_cases,
    "testd": [0] * len(obs_cases),
    "testc": [0] * len(obs_cases),
    "t": [np.nan] * len(obs_cases),
}
```

- `clinical_diagnostics` is a tuple  with all the cases reported up to time `t`. Each element consists of a tuple (x,y, disease)
where disease is either 0 (dengue), 1 (chikungunya) or 2 (other). 
- `testd` is a tuple of 0s and 1s, where 1 means that the case was tested for dengue.
- `testc` is a tuple of 0s and 1s, where 1 means that the case was tested for chikungunya.
- `t` is a tuple of the time at which each case was reported.
### Action Space

## Installation
To install the environment, run the following command in the root directory of the repository:
```
poetry install
pip install -e .
```

## Usage
To use the environment, import it as follows:
```
