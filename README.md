# Multi-Agent Deep RL for Independent Learners

### Supported features
- Quantile Network (both implicit and explicit)
- Double Learning 
- Hysteretic (supports annealing, toggled using `hynamic_h` param)
- Time Difference Likelihood!
- Risk Distortion

### Example

```bash
python main.py --env cmotp1 --likely 1 --distort_type wang
```
will train agents in `cmotp1` environment with TDL turned on and `wang` as distortion function. See more parameters in `main.py`.

### Included environments

- Meeting-In-A-Grid - `capture_target`
- CMOTP - `cmotp1, cmotp2, cmotp3`
- Climb Game - `climb_game`
- Catch-Pig - `pig`