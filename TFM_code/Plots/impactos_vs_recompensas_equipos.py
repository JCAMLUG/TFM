import pandas as pd
import matplotlib.pyplot as plt

# Se cargan los datos
df_rewards = pd.read_csv("eval_rewards.csv")
df_impactos = pd.read_csv("impactos.csv")

# Se unen ambos csv por 'Round'
df = pd.merge(df_rewards, df_impactos, on="Round")

# Se calcula totales por equipo
df["Team_1_impact"] = df["paddle_right"] + df["paddle_lower"]
df["Team_2_impact"] = df["paddle_left"] + df["paddle_upper"]
df["Team_1_reward"] = df["first_0"] + df["third_0"]
df["Team_2_reward"] = df["second_0"] + df["fourth_0"]

# Gráfica Team 1
plt.figure(figsize=(10, 5))
plt.plot(df["Round"], df["Team_1_impact"], label="Impactos", marker="o")
plt.plot(df["Round"], df["Team_1_reward"], label="Recompensas", marker="s", linestyle="--")
plt.title("Team 1 (first_0 + third_0): Impactos y Recompensas por Ronda")
plt.xlabel("Ronda")
plt.ylabel("Valor")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("team1_impactos_recompensas.png")
plt.show()

# Gráfica Team 2
plt.figure(figsize=(10, 5))
plt.plot(df["Round"], df["Team_2_impact"], label="Impactos", marker="^")
plt.plot(df["Round"], df["Team_2_reward"], label="Recompensas", marker="d", linestyle="--")
plt.title("Team 2 (second_0 + fourth_0): Impactos y Recompensas por Ronda")
plt.xlabel("Ronda")
plt.ylabel("Valor")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("team2_impactos_recompensas.png")
plt.show()
