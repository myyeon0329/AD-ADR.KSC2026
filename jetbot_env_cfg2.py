"""아이작랩 공식 오픈소스 표준 ANYmal-C 험지/계단 주행 베이스라인 환경 (jetbot_env_cfg2.py)
공식 LocomotionVelocityRoughEnvCfg를 그대로 상속받아 검증된 구조(지형, 리워드, height_scanner,
curriculum 등)를 100% 재사용하고, 여기에 Uniform Domain Randomization(UDR)만 추가한 Baseline.
"""
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


@configclass
class JetbotEnvCfg2(LocomotionVelocityRoughEnvCfg):
    """공식 구조 그대로 + Uniform Domain Randomization (Baseline)"""

    def __post_init__(self):
        super().__post_init__()

        # 로봇을 ANYmal-C로 교체
        self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ── Baseline: 매 리셋마다 균일분포로 마찰력 무작위 변경 (Uniform DR) ──
        # 공식 코드의 physics_material(startup, 고정값 0.8)은 그대로 두고,
        # 이 이벤트를 추가로 등록해서 매 에피소드 리셋마다 마찰력을 흔든다.
        self.events.randomize_friction = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.4, 1.2),
                "dynamic_friction_range": (0.3, 1.0),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )


@configclass
class JetbotEnvCfg2_EVAL(JetbotEnvCfg2):
    """레벨별 성공률 측정용: 커리큘럼 고정 + 로봇을 레벨별로 균등 배치"""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 90  # 10레벨 x 9개씩
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
