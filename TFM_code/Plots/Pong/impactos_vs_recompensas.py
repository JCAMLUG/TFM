import pandas as pd
import matplotlib.pyplot as plt

# Se cargan los datos
df_rewards = pd.read_csv("eval_rewards.csv")
df_impactos = pd.read_csv("impactos.csv")

# Se une por Round
df = pd.merge(df_rewards, df_impactos, on="Round", suffixes=("_reward", "_impact"))

# Gráfica combinada del agente de la derecha
plt.figure(figsize=(10, 5))
plt.plot(df["Round"], df["first_0_impact"], label="Impactos (first_0:)", marker="o")
plt.plot(df["Round"], df["first_0_reward"], label="Recompensa (first_0:)", marker="s")
plt.title("Agente first_0: Impactos y Recompensas por Ronda")
plt.xlabel("Ronda")
plt.ylabel("Valor")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("combinado_derecha.png")
plt.show()

# Gráfica combinada del agente de la izquierda
plt.figure(figsize=(10, 5))
plt.plot(df["Round"], df["second_0_impact"], label="Impactos (second_0)", marker="o")
plt.plot(df["Round"], df["second_0_reward"], label="Recompensa (second_0)", marker="s")
plt.title("Agente second_0: Impactos y Recompensas por Ronda")
plt.xlabel("Ronda")
plt.ylabel("Valor")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("combinado_izquierda.png")
plt.show()
