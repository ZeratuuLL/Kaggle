import pandas as pd

def join_content(x1="temp_table.pkl",x2="content_posterior.pkl"):
  temp_table = pd.read_pickle(x1)
  content_post = pd.read_pickle(x2)
  new_table = temp_table.set_index("fullVisitorId").join(content_post)
  return new_table.reset_index()
