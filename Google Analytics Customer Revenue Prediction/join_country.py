import pandas as pd

def join_country(x1="temp_table.pkl",x2="country_posterior.pkl"):
  temp_table = pd.read_pickle(x1)
  country_post = pd.read_pickle(x2)
  new_table = temp_table.set_index("country").join(country_post)
  return new_table.reset_index()
