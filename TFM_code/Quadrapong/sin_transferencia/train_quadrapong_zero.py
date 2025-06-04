import os
import json
import re
import gymnasium as gym
from pettingzoo.atari import quadrapong_v4
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from codecarbon import EmissionsTracker


# Inicio medición impacto ambiental
os.makedirs("codecarbon_logs_quadrapong_zero", exist_ok=True)

# Directorio para guardar logs de impacto ambiental
tracker = EmissionsTracker(
    project_name="pong_training",
    output_dir="codecarbon_logs_quadrapong_zero",
    log_level="ERROR"
    )

tracker.start()


STATE_FILE = "quadrapong_zero_training_state.json" # Guarda el número de ronda actual
CHECKPOINT_FREQ = 100 # Cada cuántas rondas guarda los modelos
LOG_DIR = "runs_quadrapong_zero" # Directorio de logs
MODEL_DIR = "models_quadrapong_zero" # Directorio de los modelos

# Wrapper para entrenar un agente contra los otros (modelo o aleatorio)
class QuadraAgentWrapper(gym.Env):
    def __init__(self, agent_id, opponent_models=None):
        base_env = quadrapong_v4.env(render_mode=None)

        env = ss.color_reduction_v0(base_env, mode='B') # Reducción de color a escala de grises
        env = ss.resize_v1(env, 84, 84) # Reducción de tamaño a 84x84
        env = ss.frame_stack_v1(env, 4) # Apilamiento de 4 frames

        # Se define el entorno
        self.env = env
        # Se define agente que va a entrenar
        self.agent_id = agent_id
        # Se definen los oponentes
        self.opponent_models = opponent_models or {}
        # Se define cuál es el espacio de acción
        self.action_space = env.action_space(agent_id)
        # Se define cuál es el espacio de observación
        self.observation_space = env.observation_space(agent_id)

    # Se reinicia el entorno
    def reset(self, *, seed=None, options=None):
        self.env.reset()
        # Continua hasta que sea el turno del agente
        while self.env.agent_selection != self.agent_id:
            self.env.step(0)
        # Devolver observación del agente
        obs = self.env.observe(self.agent_id)
        return obs, {}

    # Función para que el agente que estamos entrenando juega su turno, el resto de agentes son aleatorios
    def step(self, action):
        self.env.step(action) # El agente principal ejecuta una acción
        # Mientras sea el turno del oponente
        while self.env.agent_selection != self.agent_id:
            # Guarda el nombre del oponente y su estado actual
            current_agent = self.env.agent_selection
            obs, reward, terminated, truncated, info = self.env.last()

            # Si la partida terminó, el oponente ejecuta la acción 0 (quedarse quieto), esto hace que avance el turno al siguiente agente
            if terminated or truncated:
                self.env.step(0)
            else:
                # Si no ha terminado y hay modelo cargado, predice la jugada
                if current_agent in self.opponent_models:
                    act, _ = self.opponent_models[current_agent].predict(obs)
                else:
                    # Si no ha terminado y no hay modelo cargado, se elige acción aleatoria
                    act = self.env.action_space(current_agent).sample()
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
            return json.load(f).get("last_round", 0)
    return 0

# Función para cargar el último modelo actualiza
def find_latest_checkpoint(agent_id):
    # Selección del último modelo actualizado
    files = [f for f in os.listdir(MODEL_DIR) if f.startswith(f"quadrapong_agent_{agent_id}_round")]
    # Si no hay modelo devuelve None (si no hay ruta) o 0 (no hay ronda guardada)
    if not files:
        return None, 0
    # Se extrae el número de ronda
    rounds = [int(f.split('_round')[1].split('.')[0]) for f in files]
    # Localiza el modelo con ronda más alta
    max_round = max(rounds)
    # Se construye ruta completa al modelo
    path = os.path.join(MODEL_DIR, f"quadrapong_agent_{agent_id}_round{max_round}.zip")
    # Se devuelve la ruta del modelo más reciente y la ronda
    return path, max_round

# Función para cargar los modelos entrenados (checkpoints) en una ronda específica
def load_models_at_round(round_num, agents):
    # Diccionario con los modelos de cada agente
    models = {}
    # Se recorren todos los agnetes
    for agent_id in agents:
        # Carga los modelos de la ronda en custión si existen y los guarda en el diccionario
        model_path = os.path.join(MODEL_DIR, f"quadrapong_agent_{agent_id}_round{round_num}.zip")
        if os.path.exists(model_path):
            models[agent_id] = PPO.load(model_path)
        else:
            # Si no hay checkpoint para ese agente, intentar cargar último disponible
            last_path, last_round = find_latest_checkpoint(agent_id)
            if last_path:
                print(f"Cargando último checkpoint para {agent_id} en ronda {last_round}")
                models[agent_id] = PPO.load(last_path)
            # Si no encuentra checkpoints, entrena desde cero
            else:
                print(f"No se encontró checkpoint para {agent_id}, entrenando desde cero.")
                models[agent_id] = None
    return models

# Entrenamiento de los agentes
def train_quadrapong_zero(start_round=None):
    # Se crean las carpetas donde se guardarán los modelos y logs (si no existen)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Rondas de entrenamiento y pasos por ronda
    total_rounds = 2000
    timesteps_per_round = 5000

    # Se definen los 4 jugadores del juego
    quadrapong_agents = ["first_0", "second_0", "third_0", "fourth_0"]

    # Determinar desde qué ronda arrancar el entrenamiento
    if start_round is None:
        start_round = load_state()
    print(f"Reanudando desde ronda {start_round + 1}")

    # Cargar los modelos más recientes y si no hay, empieza desde cero
    trained_models = load_models_at_round(start_round, quadrapong_agents)

    # Bucle de entrenamiento
    for round_num in range(start_round + 1, total_rounds + 1):
        print(f"\nRonda {round_num}/{total_rounds}")

        # Se entrenan los 4 agentes uno por uno en cada ronda
        for agent_id in quadrapong_agents:
            print(f"Entrenando agente {agent_id}")

            # Se crea un diccionario con los modelos de los oponentes, para que el agente actual entrene contra ellos
            opponent_models = {ag: trained_models[ag] for ag in quadrapong_agents if ag != agent_id and trained_models[ag] is not None}

            # Se crea el entorno para el agente actual
            env = DummyVecEnv([lambda: QuadraAgentWrapper(agent_id, opponent_models)])
            # Se prepara el entorno para trabajar con imágenes
            env = VecTransposeImage(env)

            # Si no hay modelo previo, se crea uno nuevo
            if trained_models[agent_id] is None:
                model = PPO("CnnPolicy", env, verbose=1,

            # Si ya existe modelo, se reutiliza
            else:
                model = trained_models[agent_id]
                model.set_env(env)

            # Se entrena al agente
            model.learn(total_timesteps=timesteps_per_round)

            # Se guardar el modelo del agente (checkpoint) en la primera ronda y después cada 100 rondas
            if round_num % CHECKPOINT_FREQ == 0 or round_num == 1:
                model.save(f"{MODEL_DIR}/quadrapong_agent_{agent_id}_round{round_num}")
                print(f"Checkpoint guardado para {agent_id} en ronda {round_num}")

            # Se actualiza el diccionario con el modelo nuevo o actualizado
            trained_models[agent_id] = model

        # Se guarda en el número de ronda completada
        if round_num == 1 or round_num % CHECKPOINT_FREQ == 0:
            save_state(round_num)
            print(f"Estado guardado en ronda {round_num}")

    # Se guarda el estado final
    save_state(total_rounds)
    print("Entrenamiento multiagente Quadrapong (desde cero) completado")

# Fin medición impacto ambiental
tracker.stop()
