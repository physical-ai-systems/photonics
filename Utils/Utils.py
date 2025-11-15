import torch 
import os
import yaml


def save_data(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path+'.pt')

def load_data(path):
    return torch.load(path+'.pt')

def load_yaml(path):
    path = path + '.yaml'
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def save_yaml(yaml_file, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    path = path + '.yaml'
    with open(path, 'w') as stream:
        try:
            yaml.dump(yaml_file, stream)
        except yaml.YAMLError as exc:
            print(exc)



def arange_like(values, dim):
    '''
    This function returns a tensor with the same shape as values but with the specified dimension dim aranged from 0 to values.shape[dim]
    Input:
        values: torch.tensor
        dim: int
    Output:
        torch.tensor
    '''
    values_shape             = values.shape
    arange_dim               = torch.arange(values_shape[dim])
    arange_shape             = [1]*len(values_shape)
    arange_shape[dim]        = values_shape[dim]
    arange_target_shape      = list(values_shape)
    arange_target_shape[dim] = 1
    return arange_dim.view(arange_shape).repeat(arange_target_shape)

def select_from_values(values, dim, index):
    '''
    This function return a logical tensor with the same shape as values but with the specified dimension dim selected by the index
    Input:
        values: torch.tensor
        dim: int
        index: int
    Output:
        torch.tensor
    '''
    raise NotImplementedError

def load_data_and_config(data_path=None, config_path=None):
    if data_path is not None:
        data   = load_data(data_path)
        try:
            data_analysis = load_data(data_path+'_analysis')
            data = {**data, **data_analysis}
        except:
            pass

    else:
        data = None
        print("No data path provided")


    if config_path is not None:
        config = load_yaml(config_path)
    else:
        config = None
        print("No config path provided")

    return data, config
    
def get_data_and_config_paths(save_dir, save_name): 
            data_folder, config_folder = 'data', 'config'
            data_path, config_path = os.path.join(save_dir, data_folder, save_name), os.path.join(save_dir, config_folder, save_name)
            return data_path, config_path

 
def get_images_from_folder(folder_path):
    image_paths = []

    # Loop through the files in the folder
    for file in os.listdir(folder_path):
        # Check if the file is an image
        if file.endswith(".png") or file.endswith(".jpg"):
            image_paths.append(os.path.join(folder_path, file))

    return image_paths

def remove_from_dicts(dictionaries, keys):
    for key in keys:
        for key_dict, dictionary in dictionaries.items():
            if key in dictionary:
                if isinstance(dictionary, dict) and key in dictionary:
                    dictionary.pop(key)
                else:
                    print(f"Key {key} not found in the dictionary {key_dict}")
    return dictionaries
    
def add_to_dicts(dictionaries:dict, dict_to_add:dict):
    for key, values in dictionaries.items():
        dictionaries[key] = {**values, **dict_to_add}
    

def unsqueeze_physical_properties(list_of_physical_properties, dim):
    for i, physical_property in enumerate(list_of_physical_properties):
        physical_property.values = physical_property.values.unsqueeze(dim)
    return list_of_physical_properties

def unfold_outputs(outputs):
    '''
    Input:
        outputs: dict
        dim: int #TODO: Implement this
    Output:
        dict
    
    outputs[key] with shape (batch_size,2,...) -> outputs[key] with shape (2*batch_size,...)
    [
    [5,4,3,2,1],                               -> [0,1,2,3,4,5,6,7,8,9]
    [5,6,7,8,9]                
    ]
    '''
    for key in outputs:
        values = outputs[key].values.squeeze()
        values_shape = values.shape
        values = values.permute(1,0,*[i for i in range(2,len(values_shape))])
        values = torch.concatenate([values[0].flip(0)[:-1], values[1]], dim=0).squeeze()
        outputs[key].values = values
    return outputs

def batched_values(start, end, num_values, split_at):
    values = torch.concatenate([torch.linspace(start, split_at, num_values//2).flip(0).unsqueeze(0), torch.linspace(split_at, end, num_values//2).unsqueeze(0)], dim=0).permute(1,0)
    return values


