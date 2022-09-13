# nflfastPY

Where I keep my Python code for NFL stuff.


```python
# if I want all the data
import pandas as pd
import glob

all_files = glob.glob('/Users/justynrodrigues/Documents/nfl/data/pbp/csv/*.csv.gz')
df = pd.concat((pd.read_csv(__, low_memory=False, index_col=0) for __ in all_files))
```