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


STATE_FILE = "quadrapong_training_state.json" # Guarda el número de ronda actual
CHECKPOINT_FREQ = 100 # Cada cuántas rondas guarda los modelos
LOG_DIR = "runs_quadrapong" # Directorio de logs
MODEL_DIR = "models_quadrapong" # Directorio de los modelos

# Diccionario con el mapeo de acciones
PONG_TO_QUADRAPONG_ACTIONS = {
    0: 0,
    1: 1,
    2: 3,
    3: 4,
    4: 1,
    5: 1,
}

# Función que realiza el mapero de acciones
class ActionMappingWrapper(gym.Env):
    def __init__(self, env, mapping):
        super().__init__()
        # Se guarda el entorno base
        self.env = env
        # Se guarda el diccionario de mapeo
        self.mapping = mapping
        # Se resetea el entorno
        self.env.reset()
        # Se guarda la lista de agentes del entorno original
        self.agents = self.env.agents
        # Mapea las acciones del entorno origen
        self.action_space = {agent: gym.spaces.Discrete(len(mapping)) for agent in self.agents}
        # Se copian los espacios de observaciones del entorno base
        self.observation_space = {agent: self.env.observation_space(agent) for agent in self.agents}

    # Se reinicia el entorno
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    # Se traducen la acciones del modelo usando el diccionario de mapeo si no encuentra acción, asigna la acción 0
    def step(self, action):
        if hasattr(action, 'item'):
            action = int(action.item())
        mapped_action = self.mapping.get(action, 0)
        return self.env.step(mapped_action)

    # Los siguientes métodos hacen que el wrapper se comporte como el entorno original
    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)

# Wrapper para entrenar un agente contra los otros (modelo o aleatorio)
class QuadraAgentWrapper(gym.Env):
    def __init__(self, agent_id, opponent_models=None, pretrained_models=None):
        base_env = quadrapong_v4.env(render_mode=None)

        env = ss.color_reduction_v0(base_env, mode='B') # Reducción de color a escala de grises
        env = ss.resize_v1(env, 84, 84) # Reducción de tamaño a 84x84
        env = ss.frame_stack_v1(env, 4) # Apilamiento de 4 frames
        env = ActionMappingWrapper(env, PONG_TO_QUADRAPONG_ACTIONS) # Mapero de las acciones de Pong a Quadrapong

        # Se define el entorno
        self.env = env
        # Se define agente que va a entrenar
        self.agent_id = agent_id
        # Se definen los oponentes
        self.opponent_models = opponent_models or {}
        # Se definen los modelos preentrenados en Pong
        self.pretrained_models = pretrained_models or {}
        # Se define cuál es el espacio de acción
        self.action_space = env.action_space[agent_id]
        # Se define cuál es el espacio de observación
        self.observation_space = env.observation_space[agent_id]

    # Se reinicia el entorno
    def reset(self, *, seed=None, options=None):
        self.env.reset()
        # Continua hasta que sea el turno del agente
        while self.env.agent_selection != self.agent_id:
            self.env.step(0)
        # Devolver observación del agente
        obs = self.env.observe(self.agent_id)
        return obs, {}

    # Función para que el agente que estamos entrenando juega su turno, el re
    def step(self, action):
        self.env.step(action) # El agente principal ejecuta una acción
        # Mientras sea el turno del oponente
        while self.env.agent_selection != self.agent_id:
            # Guarda el nombre del oponente y su estado actual
            current_agent = self.env.agent_selection
            obs, reward, terminated, truncated, info = self.env.last()

            # Si la partida terminó, el oponente ejecuta la acción 0 (quedars
            if terminated or truncated:
                self.env.step(0)
            else:
                # Si no ha terminado y hay modelo cargado, predice la jugada
                if current_agent in self.opponent_models:
                    act, _ = self.opponent_models[current_agent].predict(obs)
                # Si no hay modelo entrenado, pero sí un modelo preentrenado externo, se usa ese
                elif current_agent in self.pretrained_models:
                    act, _ = self.pretrained_models[current_agent].predict(obs)
                else:
                    # Si no ha terminado y no hay modelo cargado, se elige acción aleatoria
                    act = self.env.action_space[current_agent].sample()
                # Se ejecuta la acción (predicha o aleatoria)
                self.env.step(act
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

# Función para cargar el último modelo actualizado
def find_latest_quadrapong_checkpoint():
    pattern = re.compile(r"quadrapong_agent_(first_0|second_0|third_0|fourth_0)_round(\d+)\.zip")
    rounds_found = []
    for fname in os.listdir(MODEL_DIR):
        match = pattern.match(fname)
        if match:
            rounds_found.append(int(match.group(2)))
    return min(rounds_found) if rounds_found else 0

# Se crean las carpetas donde se guardarán los modelos y logs (si no existen)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Rutas a los modelos entrenados previamente en Pong
pong_model_1_path = os.path.join("models_pong", "agent_first_round5000.zip")
pong_model_2_path = os.path.join("models_pong", "agent_second_round5000.zip")

# Se cargan los modelos previamente entrenados en Pong
pong_model_1 = PPO.load(pong_model_1_path) if os.path.exists(pong_model_1_path) else None
pong_model_2 = PPO.load(pong_model_2_path) if os.path.exists(pong_model_2_path) else None

# Se aasignan los modelos de Pong a los agentes de Quadrapong
agent_to_pong_model = {
    "first_0": pong_model_1,
    "third_0": pong_model_1,
    "second_0": pong_model_2,
    "fourth_0": pong_model_2,
}

# Se definen los 4 jugadores del juego
quadrapong_agents = ["first_0", "second_0", "third_0", "fourth_0"]

# Rondas de entrenamiento y pasos por ronda
total_rounds = 2000
timesteps_per_round = 5000
# Se carga la rond desde donde reanudar el entrenamiento
start_round = load_state()

# Si no hay modelos preentrenados, ni checkpoints, empieza entrenamiento desde cero
if start_round == 0 and (pong_model_1 is None or pong_model_2 is None):
    print("No se encontraron modelos preentrenados ni checkpoints. Comenzando entrenamiento desde cero.")
# Si hay modelos preentrenados, hace fine-tuning desde la ronda 1
elif start_round == 0 and pong_model_1 is not None and pong_model_2 is not None:
    print("Modelos preentrenados encontrados. Comenzando fine-tuning desde ronda 1.")
# Si hay checkpoints de Quadrapong, selecciona el último y continúa desde
else:
    latest_checkpoint = find_latest_quadrapong_checkpoint()
    if latest_checkpoint > 0:
        print(f"Checkpoints encontrados. Continuando entrenamiento desde ronda {latest_checkpoint + 1}")
        start_round = latest_checkpoint
    else:
        print("No se encontraron checkpoints válidos. Comenzando desde cero.")
        start_round = 0

print(f"Entrenamiento desde ronda {start_round + 1} hasta {total_rounds}")

# Bucle de entrenamiento
for round_num in range(start_round + 1, total_rounds + 1):
    print(f"\nRonda {round_num}/{total_rounds}")

    # Se entrenan los 4 agentes uno por uno en cada ronda
    for agent_id in quadrapong_agents:
        print(f"Entrenando agente {agent_id}")

        # Se crea un diccionario con los modelos de los oponentes, para que el agente actual entrene contra ellos
        # Y se carga para cada oponente el modelo más reciente
        opponent_models = {}
        for ag in quadrapong_agents:
            if ag != agent_id:
                checkpoint_path = os.path.join(MODEL_DIR, f"quadrapong_agent_{ag}_round{round_num - 1}.zip")
                if os.path.exists(checkpoint_path):
                    opponent_models[ag] = PPO.load(checkpoint_path)

        pretrained_models = {}
        # En la primera ronda, se usan los modelos preentrenados de Pong como base para los oponentes
        if round_num == 1:
            pretrained_models = {ag: agent_to_pong_model[ag] for ag in quadrapong_agents if ag != agent_id}

        # Se crea el entorno para el agente actual
        env = DummyVecEnv([lambda: QuadraAgentWrapper(agent_id, opponent_models, pretrained_models)])
        # Se prepara el entorno para trabajar con imágenes
        env = VecTransposeImage(env)

        # Se crea un nuevo modelo PPO con CNN
        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR)

        # Si existe un modelo de Pong para este agente, carga sus pesos en el modelo de Quadrapong
        if round_num == 1:
            pong_pretrained_model = agent_to_pong_model[agent_id]
            if pong_pretrained_model is not None:
                pretrained_dict = pong_pretrained_model.policy.state_dict()
                model_dict = model.policy.state_dict()
                filtered_dict = {k: v for k, v in pretrained_dict.items() if k.startswith("features_extractor.cnn")}
                model_dict.update(filtered_dict)
                model.policy.load_state_dict(model_dict)
                model.save(f"{MODEL_DIR}/quadrapong_agent_{agent_id}_pretrained")

        # Si es la primera ronda después de un reinicio, carga el último modelo guardado
        if start_round > 0 and round_num == start_round + 1:
            model_path = os.path.join(MODEL_DIR, f"quadrapong_agent_{agent_id}_round{start_round}.zip")
            if os.path.exists(model_path):
                print(f"Cargando modelo guardado para {agent_id} ronda {start_round}")
                model = PPO.load(model_path, env=env)

        # Se entrena al agente
        model.learn(total_timesteps=timesteps_per_round, log_interval=100)

        # Se guardar el modelo del agente (checkpoint) en la primera ronda y después cada 100 rondas
        if round_num == 1 or round_num % CHECKPOINT_FREQ == 0:
            model.save(f"{MODEL_DIR}/quadrapong_agent_{agent_id}_round{round_num}")
            print(f"Checkpoint guardado para {agent_id} en ronda {round_num}")

        # Se libera memoria RAM para no saturar al equipo
        del model
        for m in opponent_models.values():
            del m
        import gc
        gc.collect()

    # Se guarda en el número de ronda completada
    if round_num == 1 or round_num % CHECKPOINT_FREQ == 0:
        save_state(round_num)
        print(f"Estado guardado en ronda {round_num}")

print("Entrenamiento multiagente Quadrapong completado")

# Fin medición impacto ambiental
tracker.stop()
