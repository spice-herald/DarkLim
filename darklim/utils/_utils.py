from importlib_resources import files

cached_data_folder = files("data")

def get_cache_path(file_path):
    return str(cached_data_folder.joinpath(file_path) )
