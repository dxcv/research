from utils.dataset import DataSet


crypto_cta_config = DataSet()

crypto_cta_config.position_limit = 1
crypto_cta_config.max_gross_leverage = 1
crypto_cta_config.max_net_leverage = 1
# I might want to add asymmetric limits, i.e. different for longs and shorts
