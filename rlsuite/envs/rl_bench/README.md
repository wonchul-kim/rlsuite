

> RLBench에서 위치 값에 대한 단위는 모두 `m`(미터)

#### `ObservationConfig`

```python
class ObservationConfig(object):
    def __init__(self,
                 left_shoulder_camera: CameraConfig = None,
                 right_shoulder_camera: CameraConfig = None,
                 overhead_camera: CameraConfig = None,
                 wrist_camera: CameraConfig = None,
                 front_camera: CameraConfig = None,
                 joint_velocities=True,
                 joint_velocities_noise: NoiseModel=Identity(),
                 joint_positions=True,
                 joint_positions_noise: NoiseModel=Identity(),
                 joint_forces=True,
                 joint_forces_noise: NoiseModel=Identity(),
                 gripper_open=True,
                 gripper_pose=True,
                 gripper_matrix=False,
                 gripper_joint_positions=False,
                 gripper_touch_forces=False,
                 wrist_camera_matrix=False,
                 record_gripper_closing=False,
                 task_low_dim_state=False,
                 ):
```

사용하고자 하는 센서값을 골라서 사용할 수 있고, 카메라를 사용하게 되면 속도가 느려진다. 



```python
obs_config = ObservationConfig()
obs_config.set_all(True)

action_mode = MoveArmThenGripper(
  arm_action_mode=JointVelocity(),
  gripper_action_mode=Discrete()
)
env = Environment(action_mode, 
                  obs_config=obs_config,
                  headless=True)
env.launch()

task = env.get_task(ReachTarget)
descriptions, obs = task.reset()

for _ in range(10):
    obs, reward, terminate = task.step(np.random.normal(size=env.action_shape))

```

#### `obs`

```python

print(obs.__dict__.keys())
dict_keys([
    'left_shoulder_rgb', 'left_shoulder_depth', 'left_shoulder_mask', 'left_shoulder_point_cloud', 
    'right_shoulder_rgb', 'right_shoulder_depth', 'right_shoulder_mask', 'right_shoulder_point_cloud', 
    'overhead_rgb', 'overhead_depth', 'overhead_mask', 'overhead_point_cloud', 
    'wrist_rgb', 'wrist_depth', 'wrist_mask', 'wrist_point_cloud', 
    'front_rgb', 'front_depth', 'front_mask', 'front_point_cloud', 
    'joint_velocities', 'joint_positions', 'joint_forces', 
    'gripper_open', 'gripper_pose', 'gripper_matrix', 'gripper_joint_positions', 'gripper_touch_forces', 'task_low_dim_state', 'misc'])
```

* `task_low_dim_state`: 아래의 값들이 task에 따라서 순서대로 들어감 

    1. 목표(Target) 위치 (x, y, z) - 단위: m 

    2. 목표(Target) 방향/쿼터니언 (qx, qy, qz, qw)

    3. 경우에 따라 목표 색상, 크기, 기타 속성

    4. task별로 필요한 추가 상태값

* `joint_positions`: 로봇 팔의 각 관절에 대한 각도 (단위: radians)


* `gripper`:

    | 속성                        | 데이터 형식   | 단위            | 설명             |
    | ------------------------- | -------- | ------------- | -------------- |
    | `gripper_pose`            | (7,) 벡터  | m, quaternion | 위치 + 회전 (쿼터니언) |
    | `gripper_matrix`          | (4,4) 행렬 | m             | 위치 + 회전 (행렬)   |
    | `gripper_joint_positions` | (2,) 벡터  | rad           | finger 관절 각도   |
    | `gripper_touch_forces`    | (2,) 벡터  | N             | finger 접촉 힘    |

    * `gripper_open`: open/close에 대한 binary (close: 0, open: 1)

    * `gripper_pose`: [x, y, z, qx, qy, qz, qw]

        - x, y, z: 그리퍼 중심의 3D 위치 (m, world 좌표계)

        - qx, qy, qz, qw: 그리퍼 방향을 나타내는 쿼터니언(quaternion)

    * `gripper_matrix`: 4×4 동차변환행렬(Homogeneous transformation matrix)

        - 상위 3×3 부분: 회전 행렬(Rotation matrix)

        - 마지막 열 3개: 위치 벡터

        - 마지막 행: [0, 0, 0, 1] (행렬 형태 유지용)

        - 로봇 kinematics 계산 또는 좌표계 변환(월드 → 그리퍼 좌표계 등)

    * `gripper_joint_positions`: 보통 길이 2의 벡터 (양쪽 finger 관절 각도)

        - 각 값은 라디안 단위

        - 0에 가까울수록 완전히 닫힌 상태, 값이 커질수록 벌어진 상태

    * `gripper_touch_forces`: 길이 2의 벡터 (왼손/오른손 finger 힘)

        - 각 값은 뉴턴(N) 단위의 힘 센서 값

        - 양수면 물체나 표면과 접촉 중



