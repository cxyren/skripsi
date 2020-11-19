from streamlit import caching
import gc

print(gc.collect())
caching.clear_cache()