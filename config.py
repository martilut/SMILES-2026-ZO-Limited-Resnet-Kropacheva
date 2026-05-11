SEED = 0
SUBSET_SIZE = None # None = full dataset
HEAD_INIT = "prior" # "kaiming", "xavier", "orthogonal", "small_random", "prior"
OPTIMIZATION_MODE = "fc"
BATCH_SIZE = 128
N_BATCHES = 64
SPSA_K = 1
FREEZE = False
MOVING_AVERAGE_COEFF = 0.9
UPDATE_RULE = "momentum"   # "momentum", "adam"
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
LR = 1e-3
EPS = 1e-3
PERTURBATION_MODE = "gaussian" # "gaussian", "rademacher", "uniform"
