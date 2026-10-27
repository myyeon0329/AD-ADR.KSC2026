"""지형 난이도(레벨)별 성공률 평가 스크립트 - play.py 구조 재사용"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate success rate per terrain level.")
parser.add_argument("--num_envs", type=int, default=90, help="Number of environments (must be multiple of num_rows).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--num_episodes", type=int, default=3, help="Episodes per level to average over.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import importlib.metadata as metadata
installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ── 핵심 수정: 커리큘럼 끄고, 로봇을 지형 레벨별로 균등 배치 ──
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.curriculum = False
        num_rows = env_cfg.scene.terrain.terrain_generator.num_rows
    else:
        num_rows = 10
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.push_robot = None

    if args_cli.num_envs % num_rows != 0:
        print(f"[WARNING] num_envs({args_cli.num_envs})가 num_rows({num_rows})의 배수가 아닙니다. 정확도가 떨어질 수 있습니다.")

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    # 각 로봇을 레벨별로 균등 배치: env 0~8은 레벨0, 9~17은 레벨1 ... (num_envs=90, num_rows=10 가정)
    levels_per_env = (torch.arange(num_envs, device=device) * num_rows) // num_envs

    terrain = env.unwrapped.scene.terrain
    terrain.terrain_levels[:] = levels_per_env

    success_count = torch.zeros(num_rows)
    fail_count = torch.zeros(num_rows)

    obs = env.get_observations()
    episode_target = args_cli.num_episodes
    episodes_done_per_env = torch.zeros(num_envs, dtype=torch.long, device=device)

    max_steps = 100000
    step = 0
    while (episodes_done_per_env < episode_target).any() and step < max_steps:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

        if dones.any():
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            time_outs = extras.get("time_outs", None)
            for idx in done_ids.tolist():
                if episodes_done_per_env[idx] >= episode_target:
                    continue
                lvl = levels_per_env[idx].item()
                is_timeout = bool(time_outs[idx].item()) if time_outs is not None else False
                if is_timeout:
                    success_count[lvl] += 1
                else:
                    fail_count[lvl] += 1
                episodes_done_per_env[idx] += 1
        step += 1

    print("=" * 70)
    print(f"Task: {args_cli.task}")
    print(f"{'Level':<8}{'Success':<10}{'Fail':<8}{'SuccessRate':<12}")
    print("-" * 70)
    for lvl in range(num_rows):
        total = success_count[lvl] + fail_count[lvl]
        rate = (success_count[lvl] / total * 100) if total > 0 else float('nan')
        print(f"{lvl:<8}{int(success_count[lvl]):<10}{int(fail_count[lvl]):<8}{rate:<12.1f}")
    overall_total = success_count.sum() + fail_count.sum()
    overall_rate = success_count.sum() / overall_total * 100 if overall_total > 0 else float('nan')
    print("-" * 70)
    print(f"Overall success rate: {overall_rate:.1f}%")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
