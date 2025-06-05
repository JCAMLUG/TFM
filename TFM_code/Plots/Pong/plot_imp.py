import pandas as pd
import matplotlib.pyplot as plt

# Se cargan los datos
df = pd.read_csv("impactos.csv")

# Gráfica de impactos por agente
plt.figure(figsize=(10, 5))
plt.plot(df["Round"], df["first_0"], marker="o", label="Pala derecha (first_0)")
plt.plot(df["Round"], df["second_0"], marker="s", label="Pala izquierda (second_0)")
plt.title("Evolución de Impactos por Ronda")
plt.xlabel("Ronda")
plt.ylabel("Número de impactos")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grafica_impactos.png")
plt.show()
