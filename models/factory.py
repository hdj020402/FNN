"""Factory for creating FNN models from configuration."""
from models.fnn import FNN
from configs.schema import ModelParams


def create_model(param: ModelParams, dataset) -> FNN:
    """Create an FNN model from configuration.

    Args:
        param: ModelParams configuration.
        dataset: FeatureDataset (used to infer input/output dimensions).

    Returns:
        FNN model instance (still on CPU — caller should move to device).
    """
    hidden_layers = list(param.hidden_layer)

    # Input / output dims inferred from the dataset (which already accounts for
    # CSV features, ECFP bits, and RDKit descriptors).
    num_input = dataset.num_features
    num_output = dataset.num_targets
    dims = [num_input, *hidden_layers, num_output]
    net = FNN(dims)
    return net
