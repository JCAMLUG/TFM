import pandas as pd
import matplotlib.pyplot as plt

# Se cargan los datos
df = pd.read_csv("impactos.csv")

# Se calculan impactos por equipo
df["Team_1"] = df["paddle_right"] + df["paddle_lower"]
df["Team_2"] = df["paddle_left"] + df["paddle_upper"]

# Gráfica de impactos por equipos
plt.figure(figsize=(10, 5))
plt.plot(df["Round"], df["Team_1"], marker="o", label="Team 1 (first_0 + third_0)")
plt.plot(df["Round"], df["Team_2"], marker="s", label="Team 2 (second_0 + fourth_0)")
plt.title("Evolución de Impactos por Ronda y Equipo")
plt.xlabel("Ronda")
plt.ylabel("Número de impactos")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grafica_impactos_equipos.png")
plt.show()
