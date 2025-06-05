import pandas as pd
import matplotlib.pyplot as plt

# Se cargan los datos
df = pd.read_csv("eval_rewards.csv")

# Gráfica de recompensas por agente
plt.figure(figsize=(10, 5))
plt.plot(df["Round"], df["first_0"], marker="o", label="Agente derecha (first_0)")
plt.plot(df["Round"], df["second_0"], marker="s", label="Agente izquierda (second_0)")
plt.title("Evolución de Recompensas por Ronda")
plt.xlabel("Ronda")
plt.ylabel("Recompensa")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grafica_recompensas.png")
plt.show()
