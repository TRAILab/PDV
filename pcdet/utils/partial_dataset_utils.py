def get_nested_val(nested_dict, nested_keys):
    nested_val = nested_dict
    for key in nested_keys:
        nested_val = nested_val[key]
    return nested_val


def get_partial_image_set(cfg):
    if cfg.get('PARTIAL_IMAGE_SET'):
        with open(cfg.PARTIAL_IMAGE_SET, 'r') as image_set_file:
            partial_image_set = image_set_file.read().splitlines()
            return partial_image_set
    return []


def filter_infos(infos, partial_image_set, nested_keys):
    if len(partial_image_set) > 0:
        infos = [info for info in infos if get_nested_val(info, nested_keys) in partial_image_set]
    return infos
