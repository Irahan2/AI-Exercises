import pandas as pd
import numpy as np

length_1 = np.random.randint(800, 1600, 40)  
length_2 = np.random.randint(800, 1600, 40)  

df = pd.DataFrame({
    "length_1": length_1,
    "length_2": length_2
})

df["long"] = df["length_1"] > 1200

print(df)
