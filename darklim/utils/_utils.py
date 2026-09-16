from importlib_resources import files
import darklim as d

cached_data_folder = files(d)

def get_cache_path(file_path):
    return str(cached_data_folder.joinpath(file_path) )
