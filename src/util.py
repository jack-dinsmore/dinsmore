import os

def search_components(directory, components):
    # Find all the files in `directory` containing all of `components` in the file name
    if type(components) == str:
        components = [components]

    successful = []
    for f in os.listdir(directory):
        success = True
        for component in components:
            if not component in f:
                success = False
                break
        if success:
            successful.append(f)
    if len(successful) == 0:
        raise Exception(f"File with components {components} not found in {directory}")
    elif len(successful) > 1:
        raise Exception(f"Multiple files with components {components} found: `{successful}`")
    return f"{directory}/{successful[0]}"
