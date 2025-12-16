
import pandas as pd
import matplotlib.pyplot as plt
import sys

df = pd.read_csv(f"{sys.argv[1]}.csv")
df = df.iloc[:, 1:]

new_df = df.iloc[[0, 3]]

print(new_df)

new_df = new_df.transpose()

new_df.columns = ["load", "exectute"]

new_df.plot.bar(stacked=True, rot=0, title = f"Time for <save_location>_<execution_location> for {sys.argv[1]}")

plt.savefig(f"{sys.argv[1]}.png")