# Offline RL Module Layout

This package is split by responsibility so the pipeline is easier to maintain.

- `types.py`: data contracts and validation result models.
- `schema.py`: canonical transition schema contract.
- `validation.py`: schema/dtype/null/time-order validation.
- `io.py`: parquet loading helpers.
- `d3rlpy_adapter.py`: conversion to d3rlpy dataset object.
- `config.py`: transition builder configuration.
- `build_state_action.py`: state and action stage generation from raw data.
- `build_reward_episode.py`: reward, next_state, done, and episode-level fields.
- `transition_builder.py`: orchestration of full raw-to-transition flow.
