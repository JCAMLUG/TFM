import pandas as pd
import matplotlib.pyplot as plt

# Se cargan los datos
df = pd.read_csv("5_partidas.csv")

# Gráfica de victorias totales en 5 partidas
plt.figure(figsize=(10, 5))
plt.plot(df["Round"], df["first_0_wins"], marker="o", label="Victorias pala derecha (first_0)")
plt.plot(df["Round"], df["second_0_wins"], marker="s", label="Victorias pala izquierda (second_0)")
plt.title("Evolución de Victorias (5 Partidas)")
plt.xlabel("Ronda de entrenamiento")
plt.ylabel("Número de victorias")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grafica_victorias.png")
plt.show()
