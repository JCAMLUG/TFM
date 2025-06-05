import pandas as pd
import matplotlib.pyplot as plt

# Se cargan los datos
df = pd.read_csv("impactos.csv")

# Gráfica de impactos
plt.figure(figsize=(10, 5))
plt.plot(df["Round"], df["paddle_right"], marker="o", label="Pala derecha")
plt.plot(df["Round"], df["paddle_left"], marker="s", label="Pala izquierda")
plt.plot(df["Round"], df["paddle_lower"], marker="^", label="Pala inferior")
plt.plot(df["Round"], df["paddle_upper"], marker="d", label="Pala superior")
plt.title("Evolución de Impactos por Ronda")
plt.xlabel("Ronda")
plt.ylabel("Número de impactos")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grafica_impactos.png")
plt.show()
