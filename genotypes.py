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


