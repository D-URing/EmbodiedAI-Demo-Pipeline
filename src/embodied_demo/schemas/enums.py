from enum import Enum


class StringEnum(str, Enum):
    pass


class Difficulty(StringEnum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"


class Capability(StringEnum):
    GENERALIZATION = "generalization"
    MEMORY = "memory"
    PRECISION = "precision"
    LONG_HORIZON = "long_horizon"
    OPEN_INSTRUCTION = "open_instruction"
    STABILITY = "stability"
    SAFETY = "safety"
    EFFICIENCY = "efficiency"


class RuntimeMode(StringEnum):
    MOCK = "mock"
    REPLAY = "replay"
    SIM = "sim"
    REAL = "real"


class Launcher(StringEnum):
    LOCAL = "local"
    SLURM = "slurm"
    KUBERNETES_FUTURE = "kubernetes_future"


class PolicyTransport(StringEnum):
    INPROC = "inproc"
    WEBSOCKET = "websocket"
    GRPC_FUTURE = "grpc_future"


class ActionRepresentation(StringEnum):
    SEMANTIC = "semantic"
    JOINT = "joint"
    EE_DELTA = "ee_delta"
    HAND = "hand"


class CoordinateFrame(StringEnum):
    WORLD = "world"
    BASE = "base"
    CAMERA = "camera"
    EE = "ee"


class EnvironmentBackend(StringEnum):
    MOCK_2D = "mock_2d"
    DATASET_REPLAY = "dataset_replay"
    ROBODOJO = "robodojo"
    ROBOCASA = "robocasa"
    ROBOTWIN = "robotwin"
    SIMPLER = "simpler"
    REAL = "real"


class EvaluationProfile(StringEnum):
    SMOKE = "smoke"
    DEV = "dev"
    RELEASE = "release"
    EXTERNAL = "external"


class EvaluationLevel(StringEnum):
    E0_SCHEMA = "e0_schema"
    E1_WIRING_SMOKE = "e1_wiring_smoke"
    E2_DETERMINISTIC_MOCK = "e2_deterministic_mock"
    E3_OFFLINE_REPLAY = "e3_offline_replay"
    E4_SIMULATION = "e4_simulation"
    E5_REAL_SHADOW = "e5_real_shadow"
    E6_REAL_CLOSED_LOOP = "e6_real_closed_loop"
    E7_EXTERNAL_BENCHMARK = "e7_external_benchmark"


class FailureType(StringEnum):
    CONFIG_ERROR = "config_error"
    DEPENDENCY_ERROR = "dependency_error"
    POLICY_STARTUP_ERROR = "policy_startup_error"
    OBSERVATION_SCHEMA_ERROR = "observation_schema_error"
    ACTION_SCHEMA_ERROR = "action_schema_error"
    TRANSPORT_ERROR = "transport_error"
    TIMEOUT = "timeout"
    INVALID_ACTION = "invalid_action"
    UNSAFE_ACTION = "unsafe_action"
    TASK_FAILURE = "task_failure"
    ENVIRONMENT_INSTABILITY = "environment_instability"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    MANUAL_ABORT = "manual_abort"


class MockRealism(StringEnum):
    CONTRACT_ONLY = "contract_only"
    SYMBOLIC = "symbolic"
    KINEMATIC = "kinematic"
    PHYSICS = "physics"


class RegistryStatus(StringEnum):
    EXPERIMENTAL = "experimental"
    SUPPORTED = "supported"
    DEPRECATED = "deprecated"


class TerminationReason(StringEnum):
    SUCCESS = "success"
    TASK_FAILURE = "task_failure"
    SYSTEM_FAILURE = "system_failure"
    SAFETY_ABORT = "safety_abort"
    MAX_STEPS = "max_steps"
    MANUAL_ABORT = "manual_abort"


class VerificationStatus(StringEnum):
    UNVERIFIED = "unverified"
    REPRODUCIBLE_INTERNAL = "reproducible_internal"
    VERIFIED_INTERNAL = "verified_internal"
    VERIFIED_EXTERNAL = "verified_external"
