import gymnasium as gym
from .jetbot_env_cfg1 import JetbotEnvCfg, JetbotAdaptiveEnv
from .jetbot_env_cfg2 import JetbotEnvCfg2

AGENT_CFG_PATH = "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg"

# 1. 제안 모델 (AD-ADR + 동역학 예측기) 등록
gym.register(
    id="Isaac-Jetbot-ADADR-v0",
    entry_point=JetbotAdaptiveEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": JetbotEnvCfg,
        "rsl_rl_cfg_entry_point": AGENT_CFG_PATH,
    },
)

# 2. 공식 표준 베이스라인 (UDR) 환경 등록
gym.register(
    id="Isaac-Jetbot-Baseline-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": JetbotEnvCfg2,
        "rsl_rl_cfg_entry_point": AGENT_CFG_PATH,
    },
)


from .jetbot_env_cfg_adr_full import JetbotEnvCfgADRFull, JetbotADRFullEnv

gym.register(
    id="Isaac-Jetbot-ADRFull-v0",
    entry_point=JetbotADRFullEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": JetbotEnvCfgADRFull,
        "rsl_rl_cfg_entry_point": AGENT_CFG_PATH,
    },
)
