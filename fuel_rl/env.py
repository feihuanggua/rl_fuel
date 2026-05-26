"""Backward-compatible re-export of multi-step env (moved to legacy)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'legacy'))
from env_multistep import FuelRLEnv, FuelRLEnvSingleFrontier
