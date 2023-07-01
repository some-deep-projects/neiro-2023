import importlib
import typing as tp


def load_obj(obj_path: str, default_obj_path: str = "") -> tp.Any:
    """Extract an object from a given path.
    Taken from: https://github.com/kedro-org/kedro/blob/e78990c6b606a27830f0d502afa0f639c0830950/kedro/utils.py#L8
    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path.
    Returns:
        Extracted object.
    Raises:
        AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)
