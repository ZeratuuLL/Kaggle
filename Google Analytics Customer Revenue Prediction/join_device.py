import pandas as pd

def join_device(x1="temp_table.pkl",x2="device_posterior.pkl"):
  temp_table = pd.read_pickle(x1)
  device_post = pd.read_pickle(x2)
  new_table = temp_table.set_index("fullVisitorId").join(device_post)
  return new_table.reset_index()
