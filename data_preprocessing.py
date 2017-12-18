'''first, create a simple data frame from a csv to better grasp the problem'''

import pandas as pd
from io import StringIO
csv_data = "A,B,C,D\n1.0,2.0,3.0,4.0\n6.0,7.0,,8.0\n10.0,11.0,12.0,\n"
df = pd.read_csv(StringIO(csv_data))
print(df)


'''dealing with missing values'''