import torch 
from typing import Literal
import copy 
from torch.utils.data import Dataset, DataLoader
from Visualization.Visualization             import visualization_config
from Methods.PhysicalQuantity                import PhysicalQuantity
from Utils.Utils                             import save_data, load_data, save_yaml, load_yaml, get_data_and_config_paths, remove_from_dicts, unfold_outputs
import os
import time
import git

class StudySwipe():
    
    @staticmethod
    def get_output(study_function, parameter_dict, physical_quantity, batch_size,output_shape,main_parameter_key='wavelength', study_type='loop', save_name_suffix=''):
        for key, value in physical_quantity.items():                
            print('The study is on the parameter: ', key)

            if study_type == 'loop':
                output_all = StudySwipe.study_loop(study_function, copy.deepcopy(parameter_dict), key, value, batch_size,output_shape)
            elif study_type == 'once':
                parameter_dict[key] = copy.deepcopy(value)
                output_all = study_function(**parameter_dict, load_name=key)
            else:
                raise ValueError('The study type is not defined')

            main_parameter = copy.deepcopy(parameter_dict[main_parameter_key])
            main_parameter.set_values_in_unit()

            key_study = copy.deepcopy(value)
            key_study.set_values_in_unit()

            SaveData.save_output(output_all, main_parameter, key_study, parameter_dict['experiment_path'],save_name=key+save_name_suffix)
    
    @staticmethod   
    def study_loop(study_function, parameter_dict, key, values, batch_size, output_shape):

        dataset = Create_dataset(values.values)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, )
        len_values = len(dataset)

        for i, batch in enumerate(dataloader):
            current_batch_size = batch.shape[0]
            parameter_dict_batch               = copy.deepcopy(parameter_dict)
            parameter_dict_batch['batch_size'] = current_batch_size
            parameter_dict_batch[key]          = batch.view([current_batch_size,*[1]*len(output_shape)]).repeat([1, *output_shape]) 
            output = study_function(**parameter_dict_batch)
            print('The progress is: ', f'{((i+1)/len(dataloader))*100:.2f}%')

            for k, key_dict in enumerate(output.keys()):
                if i == 0 :
                    output_all = copy.deepcopy(output)
                    for key_dict in output.keys():
                        output_all[key_dict].values = torch.zeros([len_values, *output[key_dict].shape[1:]])

                output_all[key_dict].values[i*batch_size:min((i+1)*batch_size, len_values),...] = output[key_dict].values

        return output_all
    


class SaveData():   
    @staticmethod 
    def save_output(output_all, main_parameter, sweep_parameter, save_dir, save_name, plot_word='_plot',data_word='_data'):

        data, config = {}, {}

        main_parameter  = main_parameter[([0]*len(main_parameter.shape)-1),:] if len(main_parameter.shape)>1 else main_parameter
        sweep_parameter = sweep_parameter[([0]*len(sweep_parameter.shape)-1),:] if len(sweep_parameter.shape)>1 else sweep_parameter

        for key, value in output_all.items():
            value_len_shape = len(value.values.shape)
            value_second_dim = value.values.shape[1] if value_len_shape>1 else 1
            for i in range(value_second_dim):
                name = key + plot_word + str(i)
                x        = sweep_parameter        if value_len_shape in [1,2] else main_parameter
                x_values = sweep_parameter.values if value_len_shape in [1,2] else main_parameter.values
                y        = value                  if value_len_shape in [1,2] else sweep_parameter
                y_values = value.values           if value_len_shape == 1 else (value.values[:,i] if value_len_shape == 2 else sweep_parameter.values)
                z        = None                   if value_len_shape in [1,2] else value
                z_values = None                   if value_len_shape in [1,2] else value.values[:,i,:]

                data[name]   = SaveData.get_dict_data(x, y, x_values, y_values, z, z_values)
                config[name] = visualization_config.create_config(name)
            data[key + data_word] = {SaveData.get_label(main_parameter):main_parameter.values, SaveData.get_label(sweep_parameter):sweep_parameter.values, SaveData.get_label(value):value.values}
        
        # save the data
        data_path, config_path = get_data_and_config_paths(save_dir, save_name)
        save_data(data,    data_path)
        save_yaml(config,config_path)


    @staticmethod
    def get_dict_data(x, y, x_values, y_values, z=None, z_values=None):
        return {'x':x_values, 'y':y_values, 'x_label':SaveData.get_label(x), 'y_label':SaveData.get_label(y)} if z is None else {'x':x_values, 'y':y_values, 'z':z_values, 'x_label':SaveData.get_label(x), 'y_label':SaveData.get_label(y), 'z_label':SaveData.get_label(z)}

    @staticmethod
    def get_label(x):
        return x.name + (' (' + x.units + ')' if x.units is not None else '')

    @staticmethod
    def create_experiment_description(
                                    experiment_name, 
                                    experiment_path,
                                    experiment_description="This experiment ...",
                                    experiment_author="Ayman A. Ameen",
                                    experiment_date=time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                                    **kwargs,
                                    ):  


        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        experiment_description = {'experiment_name':experiment_name,
                                'experiment_description':experiment_description, 
                                'experiment_author':experiment_author,
                                'experiment_date':experiment_date,
                                'git_sha':sha,
                                **kwargs} 
        experiment_path = os.path.join(experiment_path, experiment_name)
        os.makedirs(experiment_path, exist_ok=True)
        save_yaml(experiment_description, os.path.join(experiment_path, 'experiment_description'))
        return experiment_path, experiment_name

class LoadData():
    @staticmethod
    def load_data(save_dir, save_name):
        data_path, config_path = get_data_and_config_paths(save_dir, save_name)
        data = load_data(data_path)
        config = load_yaml(config_path)
        return data, config

    @staticmethod
    def get_reference_data(save_dir, save_name):
        data, _ = LoadData.load_data(save_dir, save_name)
        data_name = ['Reflectancedata','Reflectance_data']
        for name in data_name:
            if name in data.keys():
                return data[name]
        

        
class Create_dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]        
