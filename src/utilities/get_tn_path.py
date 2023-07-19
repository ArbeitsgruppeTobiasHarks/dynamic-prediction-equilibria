import os


def get_tn_path() -> str:
    tn_path_from_env = os.environ.get("TNPATH")
    if tn_path_from_env is not None:
        tn_path = os.path.expanduser(tn_path_from_env)
    else:
        tn_path = os.path.expanduser("~/git/TransportationNetworks")
    if not os.path.isdir(tn_path):
        raise FileNotFoundError(
            f"Could not find the transportation networks directory at {tn_path}!"
        )
    return tn_path
