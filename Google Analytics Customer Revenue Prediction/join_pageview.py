import pandas as pd

def join_pageview(x1="temp_table.pkl",x2="pageview_posterior.pkl"):
  temp_table = pd.read_pickle(x1)
  pageview_post = pd.read_pickle(x2)
  new_table = temp_table.set_index("fullVisitorId").join(pageview_post)
  return new_table.reset_index()
