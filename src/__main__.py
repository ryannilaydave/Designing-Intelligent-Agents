from runner import Runner

runner = Runner(environments = [1,2,3], num_trials = 5)
runner.run_all_test_experiments()
