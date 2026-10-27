"""ANYmal-C 4족보행 로봇 기반 불확실성 적응형 험지/계단 주행 환경 (공식 구조 기반, 공정 비교용)"""
import torch
import torch.nn as nn
import torch.optim as optim

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


class AdvancedDynamicsPredictor(nn.Module):
    def __init__(self, state_dim=12, action_dim=12, history_len=3, latent_dim=32):
        super().__init__()
        self.history_len = history_len
        self.state_dim = state_dim

        input_dim = (state_dim + action_dim) * history_len
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_dim), nn.ReLU()
        )
        self.latent_dynamics = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, state_dim)
        )

    def forward(self, history_buffer: torch.Tensor) -> torch.Tensor:
        batch_size = history_buffer.shape[0]
        x = history_buffer.view(batch_size, -1)
        z = self.encoder(x)
        delta_z = self.latent_dynamics(z)
        delta_s = self.decoder(z + delta_z)
        current_state = history_buffer[:, -1, :self.state_dim]
        return current_state + delta_s


def randomize_terrain_friction_adaptive(env, env_ids, friction_min=0.4, friction_max=1.2, uncertainty_scale=20.0, asset_cfg=None):
    """Active Domain Randomization(Mehta et al., 2020) 원칙 적용:
    전체 범위(friction_min~friction_max)는 UDR과 완전히 동일하게 항상 고정한다.
    불확실성(U)이 높을수록, 그 고정된 범위 안에서 "중앙(쉬운 값)"보다
    "양 끝(어려운 극단값)"을 더 자주 뽑도록 샘플링 확률 분포만 조정한다.
    U가 0에 가까우면 균일분포(UDR과 동일), U가 크면 양극단에 쏠리는 U자형 분포가 된다.
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]

    num_resets = len(env_ids)
    if hasattr(env, "env_uncertainty"):
        U = env.env_uncertainty[env_ids]
    else:
        U = torch.zeros(num_resets, device=env.device)

    # U를 [0,1] 범위로 정규화 (sigmoid로 폭주 방지)
    U_norm = torch.sigmoid(uncertainty_scale * U - 2.0)

    # Beta(alpha, alpha) 분포: alpha=1이면 균일분포(UDR과 동일),
    # alpha<1이면 양 끝(0 또는 1)에 몰리는 U자형 분포가 됨
    alpha = torch.clamp(1.0 - U_norm * 0.9, min=0.1)
    beta_dist = torch.distributions.Beta(alpha, alpha)
    normalized = beta_dist.sample()  # [0,1], shape (num_resets,)

    static_friction = friction_min + normalized * (friction_max - friction_min)
    dynamic_friction = static_friction * 0.9

    env_ids_cpu = env_ids.cpu()
    materials = asset.root_physx_view.get_material_properties()
    materials[env_ids_cpu, :, 0] = static_friction.cpu().unsqueeze(-1)
    materials[env_ids_cpu, :, 1] = dynamic_friction.cpu().unsqueeze(-1)
    asset.root_physx_view.set_material_properties(materials, env_ids_cpu)


@configclass
class JetbotEnvCfg(LocomotionVelocityRoughEnvCfg):
    """제안 모델: 공식 구조 그대로 + Adaptive Dynamics-aware DR"""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.events.randomize_friction = EventTerm(
            func=randomize_terrain_friction_adaptive,
            mode="reset",
            params={
                "friction_min": 0.4,
                "friction_max": 1.2,
                "uncertainty_scale": 20.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )


class JetbotAdaptiveEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.state_dim = 12
        self.action_dim = 12
        self.history_len = 3

        self.predictor = AdvancedDynamicsPredictor(self.state_dim, self.action_dim, self.history_len).to(self.device)
        self.predictor_optimizer = optim.Adam(self.predictor.parameters(), lr=1e-3)

        self.env_uncertainty = torch.zeros(self.num_envs, device=self.device)
        self.history_buffer = torch.zeros(
            (self.num_envs, self.history_len, self.state_dim + self.action_dim),
            device=self.device
        )

    def _get_joint_pos(self):
        data = self.scene["robot"].data
        return data.joint_pos_rel if hasattr(data, "joint_pos_rel") else data.joint_pos

    def step(self, actions: torch.Tensor):
        with torch.inference_mode(False):
            with torch.no_grad():
                curr_obs = self._get_joint_pos()[:, :self.state_dim].clone()
                curr_step_data = torch.cat([curr_obs, actions.clone()[:, :self.action_dim]], dim=-1)
                self.history_buffer = torch.roll(self.history_buffer, shifts=-1, dims=1)
                self.history_buffer[:, -1, :] = curr_step_data

        obs_dict, rewards, terminated, truncated, extras = super().step(actions)

        # ── 예측기 학습: inference_mode를 완전히 벗어난 상태에서 수행 ──
        with torch.inference_mode(False):
            with torch.enable_grad():
                history_for_grad = self.history_buffer.clone()
                self.predictor_optimizer.zero_grad()

                pred_obs = self.predictor(history_for_grad)
                next_obs_real = self._get_joint_pos()[:, :self.state_dim].clone()

                loss_per_env = torch.mean((pred_obs - next_obs_real) ** 2, dim=1)
                loss = loss_per_env.mean()
                loss.backward()
                self.predictor_optimizer.step()

        alpha = 0.1
        self.env_uncertainty = (1 - alpha) * self.env_uncertainty + alpha * loss_per_env.detach()

        reset_envs = terminated | truncated
        if reset_envs.any():
            self.history_buffer[reset_envs] = 0.0

        return obs_dict, rewards, terminated, truncated, extras




