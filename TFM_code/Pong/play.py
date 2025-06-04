from pettingzoo.atari import pong_v3
import supersuit as ss
from stable_baselines3 import PPO
import time
import imageio
import os
import csv


env = pong_v3.env(render_mode="rgb_array")  # Se capturan frames como imágenes
env = ss.color_reduction_v0(env, mode='B') # Reducción de color a escala de grises
env = ss.resize_v1(env, 84, 84) # Reducción de tamaño a 84x84
env = ss.frame_stack_v1(env, 4) # Apilamiento de 4 frames

# Se define la ronda a evaluar
n_round =5000

# Rutas a los modelos entrenados
model_first_path = f"models_pong/agent_first_round{n_round}.zip"
model_second_path = f"models_pong/agent_second_round{n_round}.zip"

# Se cargan los modelos entrenados
model_first = PPO.load(model_first_path)
model_second = PPO.load(model_second_path)

# Lista vacía para almacenar los frames (imágenes) del juego
frames = []

# Se reinicia el entorno
env.reset()

# Diccionario para acumular recompensas de cada agente
rewards_accum = {"first_0": 0.0, "second_0": 0.0}

# Inicio simulación del juego
for idx, agent in enumerate(env.agent_iter()):
    # Estado del agente actual
    obs, reward, terminated, truncated, info = env.last()
    done = terminated or truncated

    # Se acumula recompensa
    if reward is not None:
        rewards_accum[agent] += reward

    # Capturar sólo 1 frame de cada 5 para acelerar 5x el video
    if idx % 5 == 0:
        frame = env.render()
        frames.append(frame)

    # Se decide la acción del agente a ejecutar
    if done:
        action = None
    else:
        if agent == "first_0":
            action, _ = model_first.predict(obs)
        else:
            action, _ = model_second.predict(obs)
    # Se ejecuta la acción
    env.step(action)
# Se cierra el entorno
env.close()

# Se crea la carpeta para guarda los videos de las partidas, si no existe
os.makedirs("videos/pong", exist_ok=True)

# Se guarda en video como mp4 con 30 fps
video_path = os.path.join("videos/pong", f"pong_first_vs_second_round{n_round}.mp4")
imageio.mimsave(video_path, frames, fps=30)
print(f"Video guardado en {video_path}")

# Se guarda o se actualiza CSV con recompensas
csv_file = "videos/pong/eval_rewards.csv"
file_exists = os.path.isfile(csv_file)

with open(csv_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        # Escribir encabezado si no existe archivo
        writer.writerow(["Round", "first_0", "second_0"])
    writer.writerow([n_round, rewards_accum["first_0"], rewards_accum["second_0"]])

# Se muestran los resultados
print(f"Recompensas guardadas en {csv_file}: {rewards_accum}")
