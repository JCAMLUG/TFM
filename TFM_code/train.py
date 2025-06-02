# Librerías
import os
import json
import gymnasium as gym
from pettingzoo.atari.pong.pong import raw_env
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from codecarbon import EmissionsTracker

# Inicio medición impacto ambiental
os.makedirs("codecarbon_logs_pog", exist_ok=True)

# Directorio para guardar logs de impacto ambiental
tracker = EmissionsTracker(
    project_name="pong_training",
    output_dir="codecarbon_logs_pog",
    log_level="ERROR"
    )

tracker.start()


STATE_FILE = "training_state.json" # Guarda el número de ronda actual
CHECKPOINT_FREQ = 100 # Cada cuántas rondas guarda los modelos
LOG_DIR = "runs_pong" # Directorio de logs
MODEL_DIR = "models_pong" # Directorio de los modelos

# Wrapper para entrenar un agente contra otro (modelo o aleatorio)
class DualAgentWrapper(gym.Env):
    def __init__(self, agent_id, opponent_model=None):
        env = raw_env()
        env = ss.color_reduction_v0(env, mode='B') # Reducción de color a escala de grises
        env = ss.resize_v1(env, 84, 84) # Reducción de tamaño a 84x84
        env = ss.frame_stack_v1(env, 4) # Apilamiento de 4 frames

        # Se define el entorno
        self.env = env
        # Se define agente que va a entrenar
        self.agent_id = agent_id
        # Se define oponente
        self.opponent_model = opponent_model
        # Se define cuál es el espacio de acción
        self.action_space = self.env.action_space(self.agent_id)
        # Se define cuál es el espacio de observación
        self.observation_space = self.env.observation_space(self.agent_id)

    # Se reinicia el entorno
    def reset(self, *, seed=None, options=None):
        self.env.reset()
        # Continua hasta que sea el turno del agente
        while self.env.agent_selection != self.agent_id:
            self.env.step(0)
        # Devolver observación del agente
        obs = self.env.observe(self.agent_id)
        return obs, {}

    # Función
    def step(self, action):
        self.env.step(action) # El agente principal ejecuta una acción
        # Mientras sea el turno del oponente
        while self.env.agent_selection != self.agent_id:
            # Guarda el nombre del oponente y su estado actual
            opponent = self.env.agent_selection
            obs, reward, terminated, truncated, info = self.env.last()
            # Si la partida terminó, el oponente ejecuta la acción 0 (quedarse quieto), esto hace que avance el turno al siguiente agente
            if terminated or truncated:
                self.env.step(0)
            else:
                # Si no ha terminado y hay modelo cargado, predice la jugada
                if self.opponent_model:
                    act, _ = self.opponent_model.predict(obs)
                else:
                    # Si no ha terminado y no hay modelo cargado, se elige acción aleatoria
                    act = self.env.action_space(opponent).sample()
                # Se ejecuta la acción (predicha o aleatoria)
                self.env.step(act)
        # Se devuelve lo que el agente principal ve
        obs, reward, terminated, truncated, info = self.env.last()
        return obs, reward, terminated, truncated, info

# Función que almacena el número de ronda en un archivo json
def save_state(round_num):
    with open(STATE_FILE, "w") as f:
        json.dump({"last_round": round_num}, f)

# Función que revisa si hay archivo json, si no lo hay, empieza entrenamiento de cero
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        return data.get("last_round", 0)
    return 0

# Si no existen, se crean los directorios para almacenar los logs y los modelos
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Rondas de entrenamiento y pasos por ronda
total_rounds = 5000
timesteps_per_round = 5000

# Se carga archivo json para saber desde que ronda se va a empezar el entrenamiento
start_round = load_state()
print(f" Resuming from round {start_round + 1}")

# Se crean los agentes
model_first = None
model_second = None

# Se carga último modelo actualizado para entrenar contra el otro, si lo hay, si no lo hay (aleatorio)
if os.path.exists(f"{MODEL_DIR}/agent_first_round{start_round}.zip"):
    model_first = PPO.load(f"{MODEL_DIR}/agent_first_round{start_round}.zip")
if os.path.exists(f"{MODEL_DIR}/agent_second_round{start_round}.zip"):
    model_second = PPO.load(f"{MODEL_DIR}/agent_first_round{start_round}.zip")

# Entrenamiento de los agentes
for round_num in range(start_round + 1, total_rounds + 1): # Continua entrenamiento desde la siguiente ronda de donde se quedó y acaba al llegar al total
    print(f"\n Round {round_num}/{total_rounds} - Training first_0 vs {'model' if model_second else 'random'}")
    # Se crea el entorno para "first_0"
    env_first = DummyVecEnv([lambda: DualAgentWrapper("first_0", model_second)])
    # Se prepara el entorno para trabajar con imágenes
    env_first = VecTransposeImage(env_first)

    # Si es la primera vez, crea el modelo con una política de CNN
    if model_first is None:
        model_first = PPO("CnnPolicy", env_first, verbose=1, tensorboard_log=LOG_DIR)
    else:
        # Si ya existe, cambia el entorno con el oponente actualizado
        model_first.set_env(env_first)

    # Se entrena "first_0" los pasos definidos
    model_first.learn(total_timesteps=timesteps_per_round, log_interval=100)


    print(f" Round {round_num}/{total_rounds} - Training second_0 vs model")
    # Se crea el entorno para "second_0"
    env_second = DummyVecEnv([lambda: DualAgentWrapper("second_0", model_first)])
    # Se prepara el entorno para trabajar con imágenes
    env_second = VecTransposeImage(env_second)

    # Si es la primera vez, crea el modelo con una política de CNN
    if model_second is None:
        model_second = PPO("CnnPolicy", env_second, verbose=1, tensorboard_log=LOG_DIR)
    else:
        # Si ya existe, cambia el entorno con el oponente actualizado
        model_second.set_env(env_second)

    # Se entrena "second_0" los pasos definidos
    model_second.learn(total_timesteps=timesteps_per_round, log_interval=100)

    # Se guardan los modelos en la primera ronda y después cada 100 rondas (checkpoints)
    if round_num == 1 or round_num % CHECKPOINT_FREQ == 0:
        model_first.save(f"{MODEL_DIR}/agent_first_round{round_num}")
        model_second.save(f"{MODEL_DIR}/agent_second_round{round_num}")
        # Se guarda el número de ronda actual en el archivo json
        save_state(round_num)
        print(f" Checkpoint saved at round {round_num}")

# Se guarda el número de ronda en el archivo json al finalizar el entrenamiento
save_state(total_rounds)
print(" Training completed.")

# Fin medición impacto ambiental
tracker.stop()
