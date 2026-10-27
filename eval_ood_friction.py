"""OOD(분포 밖) 마찰력 강건성 평가 스크립트 - 학습 때 안 본 극단적 마찰력에서 테스트"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate robustness under out-of-distribution friction.")
parser.add_argument("--num_envs", type=int, default=90)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--friction", type=float, required=True, help="Fixed OOD friction value to test (e.g. 0.1 or 2.0)")
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
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import importlib.metadata as metadata
installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.curriculum = False
        num_rows = env_cfg.scene.terrain.terrain_generator.num_rows
    else:
        num_rows = 10
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.push_robot = None

    # ── OOD 핵심: 학습 때 쓰던 마찰력 랜덤화 이벤트를 완전히 제거 ──
    # (baseline은 randomize_friction, ADADR은 randomize_terrain_friction_adaptive를 쓰는데,
    #  둘 다 여기서 꺼버리고 아래에서 강제로 고정값을 주입한다)
    env_cfg.events.randomize_friction = None

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

    levels_per_env = (torch.arange(num_envs, device=device) * num_rows) // num_envs
    terrain = env.unwrapped.scene.terrain
    terrain.terrain_levels[:] = levels_per_env

    # ── OOD 마찰력 강제 고정 함수 ──
    def force_ood_friction():
        robot = env.unwrapped.scene["robot"]
        materials = robot.root_physx_view.get_material_properties()
        materials[:, :, 0] = args_cli.friction  # static friction
        materials[:, :, 1] = args_cli.friction * 0.9  # dynamic friction
        all_ids = torch.arange(num_envs)
        robot.root_physx_view.set_material_properties(materials, all_ids)

    force_ood_friction()

    success_count = torch.zeros(num_rows)
    fail_count = torch.zeros(num_rows)

    obs = env.get_observations()
    episode_target = args_cli.num_episodes
    episodes_done_per_env = torch.zeros(num_envs, dtype=torch.long, device=device)

    max_steps = 150000
    step = 0
    while (episodes_done_per_env < episode_target).any() and step < max_steps:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

        if dones.any():
            # 리셋된 로봇들에 대해서도 OOD 마찰력을 다시 강제 적용
            force_ood_friction()

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
    print(f"Task: {args_cli.task}  |  OOD Friction: {args_cli.friction}")
    print(f"{'Level':<8}{'Success':<10}{'Fail':<8}{'SuccessRate':<12}")
    print("-" * 70)
    for lvl in range(num_rows):
        total = success_count[lvl] + fail_count[lvl]
        rate = (success_count[lvl] / total * 100) if total > 0 else float('nan')
        print(f"{lvl:<8}{int(success_count[lvl]):<10}{int(fail_count[lvl]):<8}{rate:<12.1f}")
    overall_total = success_count.sum() + fail_count.sum()
    overall_rate = success_count.sum() / overall_total * 100 if overall_total > 0 else float('nan')
    print("-" * 70)
    print(f"Overall OOD success rate (friction={args_cli.friction}): {overall_rate:.1f}%")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
