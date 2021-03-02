# Multi-Agent Deep RL for Independent Learners

## Supported Features

- DRQN (with LSTM)
- Distributed Experience Trajectories
- Quantile Networks (both implicit and explicit)
- Double Learning
- Hysteretic (supports annealing, toggled using `hynamic_h` param)
- Time Difference Likelihood!
- Risk Distortion

## Example

```bash
python main.py -n_quant 8 --env cmotp1 --likely 1 --distort_type wang
```

will train agents in `cmotp1` environment with TDL turned on and `wang` as distortion function and using 8 quantile samples. See more parameters in `main.py`.

## Included environments

- Meeting-In-A-Grid - `capture_target`
- CMOTP - `cmotp1, cmotp2, cmotp3`
- Climb Game - `climb_game`
- Catch-Pig - `pig`