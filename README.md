# Dengue_Diagnostics
This is a Gymnasium environment for learning optimal policies for acurately detecting dengue cases in the presence of 
other arbovirus cases such as chikungunya.

## About this environment
In the `DengueDiag-v0` environment, there is two epidemics going on on the same city, dengue and chikungunya. The agent
has to decide, for every case reported, whether to test for dengue or chikungunya, or not to test at all, accepting 
accepting the clinical diagnosis of the Doctor that reported the case. 
The agent receives a reward of -1 for every test performed, and a reward of episize (total number of cases reported) - the
number of cases misdiagnosed. The agent receives a reward of -10 (?) for every case that is not tested and is misdiagnosed.

At every time step, the agent receives aa list of the cases reported up to that time, and has to decide on an action for 
every new case reported at time step $t$. Assuming that the agent has already decide on an action for the cases reported before time $t$.

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
There are 6 possible actions: 
- Test for dengue (0): test the case for dengue,
- Test for chik (1): test the case for chikungunya,
- Epi confirm (2): confirm the case based on epidemiological evidence 
- Do nothing (3): do not test the case and accept the clinical diagnosis
- Confirm (4): confirm the case as a true positive
- Discard (5): discard the case as a false positive.

## Installation
To install the environment, run the following command in the root directory of the repository:
```
poetry install
pip install -e .
```

## Usage
To use the environment, import it as follows:
```
