SEED = 0
SUBSET_SIZE = None # None = full dataset
HEAD_INIT = "prior" # "kaiming", "xavier", "orthogonal", "small_random", "prior"
OPTIMIZATION_MODE = "fc_bn_all"
BATCH_SIZE = 64
N_BATCHES = 128
SPSA_K = 8
FREEZE = False
MOVING_AVERAGE_COEFF = 0.5
UPDATE_RULE = "momentum"   # "momentum", "adam",
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
LR = 1e-3
EPS = 1e-3
PERTURBATION_MODE = "gaussian" # "gaussian", "rademacher", "uniform"
