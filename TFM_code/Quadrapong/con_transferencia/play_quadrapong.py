import os
import csv
from pettingzoo.atari import quadrapong_v4
import supersuit as ss
from stable_baselines3 import PPO
import imageio

def play_quadrapong(round_num=1900): # Se define la ronda a evaluar
    env = quadrapong_v4.env(render_mode="rgb_array") # Se capturan frames como imágenes
    env = ss.color_reduction_v0(env, mode='B') # Reducción de color a escala de grises
    env = ss.resize_v1(env, 84, 84) # Reducción de tamaño a 84x84
    env = ss.frame_stack_v1(env, 4) # Apilamiento de 4 frames

    # Rutas a los modelos entrenados
    model_paths = {
        "first_0": f"models_quadrapong/quadrapong_agent_first_0_round{round_num}.zip",
        "second_0": f"models_quadrapong/quadrapong_agent_second_0_round{round_num}.zip",
        "third_0": f"models_quadrapong/quadrapong_agent_third_0_round{round_num}.zip",
        "fourth_0": f"models_quadrapong/quadrapong_agent_fourth_0_round{round_num}.zip",
    }

    # Se cargan los modelos entrenados
    models = {}
    for agent, path in model_paths.items():
        if os.path.exists(path):
            models[agent] = PPO.load(path)
        else:
            print(f"Modelo no encontrado para {agent}: {path}")

    # Lista vacía para almacenar los frames (imágenes) del juego
    frames = []
    # Diccionario para acumular recompensas de cada agente
    rewards_accum = {agent: 0.0 for agent in model_paths.keys()}
    # Se reinicia el entorno
    env.reset()

    # Inicio simulación del juego
    for idx, agent in enumerate(env.agent_iter()):
        # Estado del agente actual
        obs, reward, terminated, truncated, info = env.last()
        done = terminated or truncated

        # Se acumula recompensa
        if reward is not None:
            rewards_accum[agent] += reward

        # Capturar sólo 1 frame de cada 10 para acelerar 10x el video
        if idx % 10 == 0:
            frame = env.render()
            frames.append(frame)

        # Se decide la acción del agente a ejecutar
        if done:
            action = None
        else:
            if agent in models:
                action, _ = models[agent].predict(obs)
            else:
                action = env.env.action_space(agent).sample()
        # Se ejecuta la acción
        env.step(action)
    # Se cierra el entorno
    env.close()

    # Se crea la carpeta para guarda los videos de las partidas, si no existe
    os.makedirs("videos/quadrapong", exist_ok=True)

    # Se guarda en video como mp4 con 30 fps
    video_path = os.path.join("videos/quadrapong", f"quadrapong_round{round_num}.mp4")
    imageio.mimsave(video_path, frames, fps=30)
    print(f"Video guardado en {video_path}")

    # Se guarda o se actualiza CSV con recompensas
    csv_file = "videos/quadrapong/eval_rewards.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Round", "first_0", "second_0", "third_0", "fourth_0"])
        writer.writerow([
            round_num,
            rewards_accum["first_0"],
            rewards_accum["second_0"],
            rewards_accum["third_0"],
            rewards_accum["fourth_0"],
        ])
    # Se muestran los resultados
    print(f"Recompensas guardadas en {csv_file}: {rewards_accum}")

if __name__ == "__main__":
    play_quadrapong()
