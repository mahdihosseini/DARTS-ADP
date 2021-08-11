from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

DARTS_ADP_N2 = Genotype(
  normal=[
    ('max_pool_3x3', 1), 
    ('max_pool_3x3', 0), 
    ('dil_conv_5x5', 2), 
    ('max_pool_3x3', 1)
  ], 
  normal_concat=range(2, 4), 
  reduce=[
    ('sep_conv_5x5', 0), 
    ('max_pool_3x3', 1), 
    ('max_pool_3x3', 2), 
    ('dil_conv_5x5', 0)
  ], 
  reduce_concat=range(2, 4)
)

DARTS_ADP_N3 = Genotype(
  normal=[
    ('max_pool_3x3', 0), 
    ('max_pool_3x3', 1), 
    ('sep_conv_5x5', 2), 
    ('max_pool_3x3', 1), 
    ('dil_conv_5x5', 3), 
    ('max_pool_3x3', 1)
  ], 
  normal_concat=range(2, 5), 
  reduce=[
    ('max_pool_3x3', 0), 
    ('dil_conv_5x5', 1), 
    ('max_pool_3x3', 0), 
    ('max_pool_3x3', 2), 
    ('skip_connect', 1), 
    ('max_pool_3x3', 0)
  ], 
  reduce_concat=range(2, 5)
)

DARTS_ADP_N4 = Genotype(
  normal=[
    ('max_pool_3x3', 0), 
    ('max_pool_3x3', 1), 
    ('max_pool_3x3', 0), 
    ('skip_connect', 2), 
    ('max_pool_3x3', 0), 
    ('max_pool_3x3', 2), 
    ('dil_conv_3x3', 4), 
    ('max_pool_3x3', 0)
  ], 
  normal_concat=range(2, 6), 
  reduce=[
    ('max_pool_3x3', 0), 
    ('max_pool_3x3', 1), 
    ('dil_conv_5x5', 2), 
    ('max_pool_3x3', 0), 
    ('sep_conv_5x5', 2), 
    ('max_pool_3x3', 0), 
    ('dil_conv_3x3', 2), 
    ('max_pool_3x3', 4)
  ], 
  reduce_concat=range(2, 6)
)

#### adp, 4 cell
# adas 4 cell
# DARTS_adas_adp_175_16_98_n_4_c_4_seed_0 = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('max_pool_3x3', 0), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('dil_conv_3x3', 2), ('skip_connect', 1), ('sep_conv_5x5', 4), ('dil_conv_3x3', 0)], reduce_concat=range(2, 6))
DARTS_adas_adp_175_16_98_n_4_c_4_seed_1 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('dil_conv_3x3', 4), ('max_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 2), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))
# DARTS_adas_adp_175_16_98_n_4_c_4_seed_2 = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 3), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('max_pool_3x3', 4), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))
# DARTS_adas_adp_175_16_98_n_4_c_4_seed_3 = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('dil_conv_5x5', 3), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 2), ('sep_conv_3x3', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

#### adp, 4 cell 2 node
# darts 4 cell 2 node
# DARTS_adp_175_16_n_2_c_4_size_64_seed_0 = Genotype(normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2)], normal_concat=range(2, 4), reduce=[('skip_connect', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 0)], reduce_concat=range(2, 4))
# DARTS_adp_175_16_n_2_c_4_size_64_seed_1 = Genotype(normal=[('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 0)], normal_concat=range(2, 4), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 0)], reduce_concat=range(2, 4))
# DARTS_adp_175_16_n_2_c_4_size_64_seed_2 = Genotype(normal=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 1)], normal_concat=range(2, 4), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 4))
# adas 4 cell 2 node
DARTS_adas_adp_175_16_98_n_2_c_4_seed_0 = Genotype(normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 1)], normal_concat=range(2, 4), reduce=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('dil_conv_5x5', 0)], reduce_concat=range(2, 4))
# DARTS_adas_adp_175_16_98_n_2_c_4_seed_1 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0)], normal_concat=range(2, 4), reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 2)], reduce_concat=range(2, 4))
# DARTS_adas_adp_175_16_98_n_2_c_4_seed_2 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 4), reduce=[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 1)], reduce_concat=range(2, 4))
#### adp, 4 cell 3 node
# darts 4 cell 3 node
# DARTS_adp_175_16_n_3_c_4_size_64_seed_0 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('dil_conv_5x5', 3), ('max_pool_3x3', 0)], normal_concat=range(2, 5), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 2), ('max_pool_3x3', 3)], reduce_concat=range(2, 5))
# DARTS_adp_175_16_n_3_c_4_size_64_seed_1 = Genotype(normal=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3)], normal_concat=range(2, 5), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('skip_connect', 0), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2)], reduce_concat=range(2, 5))
# DARTS_adp_175_16_n_3_c_4_size_64_seed_2 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('max_pool_3x3', 2)], normal_concat=range(2, 5), reduce=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('sep_conv_5x5', 1), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2)], reduce_concat=range(2, 5))
# adas 4 cell 3 node
DARTS_adas_adp_175_16_98_n_3_c_4_seed_0 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 1), ('dil_conv_5x5', 3), ('max_pool_3x3', 1)], normal_concat=range(2, 5), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('skip_connect', 1), ('max_pool_3x3', 0)], reduce_concat=range(2, 5))
# DARTS_adas_adp_175_16_98_n_3_c_4_seed_1 = Genotype(normal=[('max_pool_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_5x5', 0)], normal_concat=range(2, 5), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('max_pool_3x3', 2)], reduce_concat=range(2, 5))
# DARTS_adas_adp_175_16_98_n_3_c_4_seed_2 = Genotype(normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('max_pool_3x3', 1)], normal_concat=range(2, 5), reduce=[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 2), ('sep_conv_3x3', 3)], reduce_concat=range(2, 5))

