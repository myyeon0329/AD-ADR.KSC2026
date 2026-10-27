"""ADR (Active Domain Randomization) - Mehta et al. 2020 완전 구현
- 지형/초기위치 쌍 동기화: env_i (randomized)와 env_{i+half} (reference)가 동일 지형
- 리셋 동기화: env_i가 리셋되면 env_{i+half}도 강제로 함께 리셋
- 정책 학습 배제: PPO는 randomized(앞 half)만 학습, reference는 판별기 학습에만 사용
  (실제 배제는 adr_filtered_wrapper.py의 ADRFilteredVecEnvWrapper가 담당)
"""
import torch
import torch.nn as nn
import torch.optim as optim

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG
from isaaclab_tasks.manager_based.jetbot_dr.svpg_sampler import SVPGSampler

NOMINAL_FRICTION = 0.8
FRICTION_MIN, FRICTION_MAX = 0.4, 1.2
NUM_PARTICLES = 5


class Discriminator(nn.Module):
    def __init__(self, state_dim=12, action_dim=12):
        super().__init__()
        input_dim = state_dim * 2 + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, s_t, a_t, s_t1):
        x = torch.cat([s_t, a_t, s_t1], dim=-1)
        return self.net(x)


def randomize_friction_adr(env, env_ids, asset_cfg=None):
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    device = env.device
    half = env.half

    # ── 리셋 동기화: randomized(앞쪽) 환경이 리셋되면, 짝인 reference(뒤쪽) 환경도
    # 강제로 같이 리셋되도록 표시. 무한 재귀 방지를 위해 플래그 사용. ──
    is_ref_ids = env.is_reference[env_ids]
    randomized_ids_in_batch = env_ids[~is_ref_ids]
    if len(randomized_ids_in_batch) > 0 and not getattr(env, "_syncing_pair", False):
        paired_ref_ids = randomized_ids_in_batch + half
        paired_ref_ids = paired_ref_ids[paired_ref_ids < env.num_envs]
        if len(paired_ref_ids) > 0:
            env._syncing_pair = True
            try:
                env._reset_idx(paired_ref_ids)
            finally:
                env._syncing_pair = False

    num_resets = len(env_ids)
    is_ref = env.is_reference[env_ids]
    friction = torch.full((num_resets,), NOMINAL_FRICTION, device=device)

    rand_mask = ~is_ref
    if rand_mask.any():
        num_rand = rand_mask.sum().item()
        particle_idx = torch.randint(0, NUM_PARTICLES, (num_rand,), device=device)
        normalized = env.current_proposals[particle_idx, 0]
        friction[rand_mask] = FRICTION_MIN + normalized * (FRICTION_MAX - FRICTION_MIN)
        env.env_particle_assignment[env_ids[rand_mask]] = particle_idx

    dynamic_friction = friction * 0.9
    env_ids_cpu = env_ids.cpu()
    materials = asset.root_physx_view.get_material_properties()
    materials[env_ids_cpu, :, 0] = friction.cpu().unsqueeze(-1)
    materials[env_ids_cpu, :, 1] = dynamic_friction.cpu().unsqueeze(-1)
    asset.root_physx_view.set_material_properties(materials, env_ids_cpu)

    # ── 지형 동기화: 리셋 후에도 항상 짝(pair)이 같은 지형을 유지하도록 강제 ──
    terrain = env.scene.terrain
    terrain.terrain_levels[half:] = terrain.terrain_levels[:half]
    terrain.terrain_types[half:] = terrain.terrain_types[:half]
    terrain.env_origins[half:] = terrain.env_origins[:half]


@configclass
class JetbotEnvCfgADRFull(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.events.randomize_friction = EventTerm(
            func=randomize_friction_adr,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot")},
        )


class JetbotADRFullEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.state_dim = 12
        self.action_dim = 12
        self.half = self.num_envs // 2

        # 고정 그룹 배정: 앞 half = randomized, 뒤 half = reference
        self.is_reference = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_reference[self.half:] = True

        # 최초 1회 지형 동기화
        terrain = self.scene.terrain
        terrain.terrain_levels[self.half:] = terrain.terrain_levels[: self.half]
        terrain.terrain_types[self.half:] = terrain.terrain_types[: self.half]
        terrain.env_origins[self.half:] = terrain.env_origins[: self.half]

        self.svpg = SVPGSampler(num_particles=NUM_PARTICLES, param_dim=1, device=str(self.device))
        self.current_proposals = self.svpg.propose()
        try:
            import os as _os
            with open("adr_prop_" + _os.environ.get("ADR_LOG_TAG", "run") + ".txt", "a") as _f:
                _f.write(str(getattr(self, "_step_counter", 0)) + " " + " ".join(f"{v:.4f}" for v in self.current_proposals[:, 0].tolist()) + "\n")
        except Exception:
            pass

        self.discriminator = Discriminator(self.state_dim, self.action_dim).to(self.device)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)

        self.env_particle_assignment = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._svpg_update_interval = 1000
        self._step_counter = 0

        self._env_disc_reward_accum = torch.zeros(self.num_envs, device=self.device)
        self._env_disc_reward_count = torch.zeros(self.num_envs, device=self.device)
        self._particle_completed_scores = [[] for _ in range(NUM_PARTICLES)]

    def _get_joint_pos(self):
        data = self.scene["robot"].data
        return data.joint_pos_rel if hasattr(data, "joint_pos_rel") else data.joint_pos

    def step(self, actions):
        s_t = self._get_joint_pos()[:, : self.state_dim].clone()

        obs_dict, rewards, terminated, truncated, extras = super().step(actions)

        with torch.inference_mode(False):
            with torch.enable_grad():
                s_t1 = self._get_joint_pos()[:, : self.state_dim].clone()
                a_t = actions[:, : self.action_dim].clone()

                labels = (~self.is_reference).float().unsqueeze(-1)
                logits = self.discriminator(s_t, a_t, s_t1)
                self.disc_optimizer.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                loss.backward()
                self.disc_optimizer.step()

                with torch.no_grad():
                    disc_reward = torch.sigmoid(logits).squeeze(-1)
                    self._env_disc_reward_accum += disc_reward
                    self._env_disc_reward_count += 1

                    done_mask = terminated | truncated
                    if done_mask.any():
                        rand_done = done_mask & (~self.is_reference)
                        if rand_done.any():
                            traj_scores_rand = self._env_disc_reward_accum[rand_done] / self._env_disc_reward_count[rand_done].clamp(min=1)
                            particles_done = self.env_particle_assignment[rand_done]
                            for p in range(NUM_PARTICLES):
                                pmask = particles_done == p
                                if pmask.any():
                                    self._particle_completed_scores[p].extend(traj_scores_rand[pmask].tolist())
                        self._env_disc_reward_accum[done_mask] = 0.0
                        self._env_disc_reward_count[done_mask] = 0.0

        self._step_counter += 1
        if self._step_counter % self._svpg_update_interval == 0:
            with torch.inference_mode(False):
                with torch.enable_grad():
                    avg_rewards = torch.zeros(NUM_PARTICLES, device=self.device)
                    for p in range(NUM_PARTICLES):
                        scores = self._particle_completed_scores[p]
                        avg_rewards[p] = torch.tensor(scores, device=self.device).mean() if len(scores) > 0 else 0.0
                        self._particle_completed_scores[p] = []
                    try:
                        import os as _os
                        _r = [float(x) for x in torch.as_tensor(avg_rewards).flatten().tolist()]
                        with open("adr_rew_" + _os.environ.get("ADR_LOG_TAG", "run") + ".txt", "a") as _f:
                            _f.write(str(self._step_counter) + " " + " ".join(f"{x:.5f}" for x in _r) + "\n")
                    except Exception:
                        pass
                    self.svpg.update(avg_rewards)
                    self.current_proposals = self.svpg.propose()
                    try:
                        import os as _os
                        with open("adr_prop_" + _os.environ.get("ADR_LOG_TAG", "run") + ".txt", "a") as _f:
                            _f.write(str(getattr(self, "_step_counter", 0)) + " " + " ".join(f"{v:.4f}" for v in self.current_proposals[:, 0].tolist()) + "\n")
                    except Exception:
                        pass

        return obs_dict, rewards, terminated, truncated, extras
