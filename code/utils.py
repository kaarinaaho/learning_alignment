import numpy as np


def get_95_ci(df):
    """Add columns for upper and lower bounds of 95% CI to input df."""
    print(df)
    df['upperci'] = df.apply(
        lambda x: (
            x["mean_correct"] + 1.96 * x["std_correct"]/np.sqrt(x["n_pid"])
            ), axis=1
        )
    df["lowerci"] = df.apply(
        lambda x: (
            x["mean_correct"] - 1.96 * x["std_correct"]/np.sqrt(x["n_pid"])
            ), axis=1
        )

    return df
