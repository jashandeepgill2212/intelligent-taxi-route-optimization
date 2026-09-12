import os
import json
import threading
from typing import Dict, Any, Optional
from ml.graph.builder import RoadNetworkGraph
from ml.evaluation.baselines import BaselineRouter
from ml.evaluation.compare import AlgorithmEvaluator
from ml.training.train_dqn import train_dqn


class RoutingService:
    """
    Singleton service holding shared graph instance, algorithm evaluator,
    and background training job state.
    """
    _instance: Optional['RoutingService'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RoutingService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.checkpoint_dir = os.path.abspath("ml/checkpoints")
        self.model_path = os.path.join(self.checkpoint_dir, "dqn_model.pt")
        self.metrics_path = os.path.join(self.checkpoint_dir, "training_metrics.json")
        
        self.rng = RoadNetworkGraph()
        # Build synthetic grid network as primary fast loader
        self.rng.build_synthetic_grid()
        self.current_traffic = "MEDIUM"
        self.rng.set_traffic_condition(self.current_traffic)

        self.baseline_router = BaselineRouter(self.rng)
        self.evaluator = AlgorithmEvaluator(self.rng, dqn_model_path=self.model_path)

        # Training job state
        self.is_training = False
        self.training_thread: Optional[threading.Thread] = None
        self.training_status = "idle"  # idle, training, completed, failed
        self.training_progress = {"episodes_total": 0, "current_episode": 0, "message": "Not started"}

    def set_traffic_level(self, traffic_level: str):
        self.current_traffic = traffic_level.upper()
        self.rng.set_traffic_condition(self.current_traffic)
        # Re-initialize evaluator with updated graph weights
        self.evaluator = AlgorithmEvaluator(self.rng, dqn_model_path=self.model_path)

    def get_graph_status(self) -> Dict[str, Any]:
        return {
            'city_name': self.rng.city_name,
            'is_synthetic': self.rng.is_synthetic,
            'num_nodes': self.rng.graph.number_of_nodes() if self.rng.graph else 0,
            'num_edges': self.rng.graph.number_of_edges() if self.rng.graph else 0,
            'current_traffic_level': self.current_traffic
        }

    def start_background_training(self, episodes: int = 150, batch_size: int = 32):
        if self.is_training:
            return False, "Training is already in progress."

        self.is_training = True
        self.training_status = "training"
        self.training_progress = {"episodes_total": episodes, "current_episode": 0, "message": "Training started"}

        def _run_train():
            try:
                train_dqn(
                    episodes=episodes,
                    batch_size=batch_size,
                    checkpoint_path=self.model_path,
                    metrics_path=self.metrics_path
                )
                self.training_status = "completed"
                self.training_progress["message"] = "Training completed successfully."
                # Reload evaluator with new model
                self.evaluator = AlgorithmEvaluator(self.rng, dqn_model_path=self.model_path)
            except Exception as e:
                self.training_status = "failed"
                self.training_progress["message"] = f"Training error: {str(e)}"
            finally:
                self.is_training = False

        self.training_thread = threading.Thread(target=_run_train, daemon=True)
        self.training_thread.start()
        return True, "Training process launched in background."

    def get_training_status(self) -> Dict[str, Any]:
        metrics = {}
        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, 'r') as f:
                    metrics = json.load(f)
            except Exception:
                pass

        return {
            'status': self.training_status,
            'is_training': self.is_training,
            'progress': self.training_progress,
            'model_checkpoint_exists': os.path.exists(self.model_path),
            'metrics': metrics
        }
