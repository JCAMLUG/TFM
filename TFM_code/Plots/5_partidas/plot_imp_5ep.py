import pandas as pd
import matplotlib.pyplot as plt

# Se cargan los datos
df = pd.read_csv("5_partidas.csv")

# Gráfica de impactos totales  en 5 partidas
plt.figure(figsize=(10, 5))
plt.plot(df["Round"], df["hit_first_0"], marker="o", label="Impactos pala derecha (first_0)")
plt.plot(df["Round"], df["hit_second_0"], marker="s", label="Impactos pala izquierda (second_0)")
plt.title("Evolución de Impactos Totales (5 Partidas)")
plt.xlabel("Ronda de entrenamiento")
plt.ylabel("Número total de impactos")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grafica_impactos_totales.png")
plt.show()
