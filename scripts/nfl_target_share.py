
import pandas as pd

df = pd.DataFrame()

target_df = (
    df.groupby(
        ["game_id", "receiver_player_id", "receiver_player_name", "posteam"], as_index=False,
    )["pass_attempt"].sum()
    .merge(
        df.loc[df["receiver_player_id"].notnull()]
        .groupby(["game_id", "posteam"], as_index=False
        )["pass_attempt"].sum(),
        on=["game_id", "posteam"], suffixes=("_receiver", "_team"),
    ))

target_df["target_share"] = (target_df["pass_attempt_player"] / target_df["pass_attempt_team"])

target_df = target_df.groupby(["receiver_player_id", "receiver_player_name", "posteam"], as_index=False).mean()

target_df[["receiver_player_name", "target_share"]].sort_values(by="target_share", ascending=False).head()