import pandas as pd
import matplotlib.pyplot as plt

# Se cargan los datos
df = pd.read_csv("eval_rewards.csv")

# Gráfica de recompensas 
plt.figure(figsize=(10, 5))
plt.plot(df["Round"], df["first_0"], marker="o", label="Team 1")
plt.plot(df["Round"], df["second_0"], marker="s", label="Team 2")
plt.title("Evolución de Recompensas por Ronda y Equipos")
plt.xlabel("Ronda")
plt.ylabel("Recompensa")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig("grafica_recompensas_por_equipos.png")
plt.show()
