import d3rlpy
import numpy as np

def test_evaluator_signature():
    # Create dummy data
    obs = np.random.random((100, 11)).astype(np.float32)
    actions = np.random.randint(0, 5, size=(100, 1)).astype(np.int64)
    rewards = np.random.random((100, 1)).astype(np.float32)
    terminals = np.zeros((100,), dtype=np.float32)
    
    dataset = d3rlpy.dataset.MDPDataset(
        observations=obs,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )
    
    config = d3rlpy.algos.DiscreteCQLConfig()
    algo = config.create()
    algo.build_with_dataset(dataset)
    
    class TraceEvaluator:
        def __call__(self, algo, dataset):
            print(f"Evaluator called with dataset type: {type(dataset)}")
            return 0.0
            
    try:
        algo.fit(
            dataset,
            n_steps=1,
            n_steps_per_epoch=1,
            evaluators={"trace": TraceEvaluator()},
            show_progress=False
        )
    except Exception as e:
        print(f"Fit failed: {e}")

if __name__ == "__main__":
    test_evaluator_signature()
