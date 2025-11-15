# Importing common libraries
import os                         # OS module in Python provides functions for interacting with the operating system
import numpy as np                 # NumPy is a Python library used for working with arrays
import torch
from Utils.Utils import save_data, load_data, load_yaml, save_yaml, load_data_and_config, get_data_and_config_paths, remove_from_dicts, add_to_dicts
import plotly.graph_objects as go  # Plotly Graph Objects is the low-level interface to figures, traces and layout in Plotly.py


# Pooling 
def pool_data(data, pool_size):
    '''
    data: list of pytorch tensors
    pool_size: int
    '''
    for i in range(len(data)):

        data_size = len(data[i].shape)
        # pooling operation accept [batch_size, channels, height, width] or [batch_size, channels, length]
        # check if the shape of the data equals 2 or 1 
        if data_size == 2: # Do the pool only on the last dimension
            data[i] = data[i].unsqueeze(0) # [1, height, width]
        elif data_size == 1:
            data[i] = data[i].unsqueeze(0).unsqueeze(0) # [1, 1, length]
        else:
            raise ValueError('The shape of the data must be 2 or 1')
        
        data[i] = torch.nn.functional.avg_pool1d(data[i], pool_size)
        data[i] = data[i].squeeze(0).squeeze(0)

    return data

def torch_to_numpy(tensor):
    if type(tensor) == torch.Tensor:
        return tensor.detach().cpu().numpy()
    else:
        return tensor

def prepare_data(df = None, # dataframe
                    x  = None , # x axis
                    y  = None, # y axis/axes
):
    assert x is not None, 'x must be provided'
    assert y is not None, 'y must be provided'

    if df is not None:
        x = df[x]
        y = df[y]    
    elif type(y) == list:
        x = torch_to_numpy(x)
        y = [torch_to_numpy(y_) for y_ in y]
    elif type(y) == torch.Tensor or type(y) == np.ndarray:
        x = torch_to_numpy(x)
        if len(y.shape) == 1:
            assert len(y) == len(x), 'x and y must have the same length'
            y = torch_to_numpy(y)
            y = [y]
        elif len(y.shape) == 2:
            assert y.shape[-1] == len(x), 'x and y must have the same length'
            y = [torch_to_numpy(y[counter,...]).flatten() for counter in range(y.shape[0])]
        else:
            raise ValueError('y must be a 1D or 2D array')
    else:
        raise ValueError('y must be a list, a torch.Tensor or a np.ndarray')
    return x, y

def save_image(fig, save_dir, save_name, save_format, width, height):
    save_dir_images, save_dir_html = os.path.join(save_dir, 'images'), os.path.join(save_dir, 'html')
    os.makedirs(save_dir_images, exist_ok=True)
    os.makedirs(save_dir_html, exist_ok=True)

    for i in range(len(save_format)):
        fig.write_image( os.path.join(save_dir_images, save_name + '.' + save_format[i]),
                        scale=5, # Increase the resolution of the figure
                        width=width, # width of the figure
                        height=height, # height of the figure
                            )    
    fig.write_html(os.path.join(save_dir_html, save_name + '.html'))

def plot(df = None, # dataframe
           x  = None , # x axis
           y  = None, # y axis/axes
           names = None, # names of the traces
           highlight_points = None, # points to highlight
           title = None, x_label = None, y_label = None, # title and labels
           line_width = 2, line_colors = ['royalblue', 'firebrick', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'black'], # line colors
           line_dash = ['solid', 'dash', 'dashdot', 'dot', 'longdash', 'longdashdot', 'longdashdotdot', 'solid', 'dash', 'dashdot', 'dot'],  # line dash                                                
           marker_colors = ['royalblue', 'firebrick', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'black'], # marker colors
           marker_sizes = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], # marker sizes
           xlim = None, ylim = None, # x and y limits
           mode = 'lines', # mode of the trace (lines, markers, lines+markers) 
           opacity = 0.7, # opacity of the trace
           width = 600, height = 400, # width and height of the figure
           template = 'plotly_white', # template of the figure (plotly, plotly_white, plotly_dark, ggplot2, seaborn, simple_white, none)
           legend_orientation = 'v', # legend orientation (h = horizontal, v = vertical)
           legend_x = 0.5, legend_y = -0.1, # legend position (x = 0.5, y = -0.2)
           plot_bgcolor = 'white', # background color of the plot
           save_dir = None, # directory to save the figure
           save_name = 'figure', # name of the figure
           save_format = ['png'],# format of the figure (png, jpeg, pdf, svg, webp)
           show = True # show the figure
           ):
    
    # Preparing the data
    x, y = prepare_data(df, x, y)  

    # Creating the figure
    fig = go.Figure()

    # Adding traces to the figure
    for i in range(len(y)):
        # Adding traces to the figure (lines, markers, lines+markers) - The trace type can be line charts, scatter charts,text charts, and bubble charts
        fig.add_trace(
            go.Scatter(x=x,
                        y=y[i],
                        name=names[i] if names is not None else None,
                        line=dict(
                            dash=line_dash[i],
                            color=line_colors[i],
                            width=line_width,
                        ),
                       marker=dict(    
                            color=marker_colors[i],
                            size=marker_sizes[i],
                        ),
            )
        )
    if highlight_points is not None:
        for i in range(len(highlight_points)):
            if type(highlight_points[i]) == list:
                for j in range(len(highlight_points[i])):
                    assert highlight_points[i][j] < len(x), 'highlight_points must be less than the length of x'
                    assert highlight_points[i][j] >= 0, 'highlight_points must be greater than or equal to 0'
                    assert type(highlight_points[i][j]) == int, 'highlight_points must be an integer'
                    fig.add_trace(
                        go.Scatter(x=[x[highlight_points[i][j]]],
                                y=[y[i][highlight_points[i][j]]],
                                mode='markers',
                                    marker=dict(
                                    color=marker_colors[i],
                                    size=marker_sizes[i],
                                    symbol = 'star',
                                    ),
                                    ),
                        )
            elif type(highlight_points[i]) is dict:
                for key, value in highlight_points[i].items():
                    assert value < len(x), 'highlight_points must be less than the length of x'
                    assert value >= 0, 'highlight_points must be greater than or equal to 0'
                    assert type(value) == int, 'highlight_points must be an integer'
                    fig.add_trace(
                        go.Scatter(x=[x[value]],
                                y=[y[i][value]],
                                name= names[i] + ' ' + key if names[i] is not None else key,
                                mode='markers',
                                    marker=dict(
                                    color=marker_colors[i],
                                    size=marker_sizes[i],
                                    symbol = 'star',
                                    ),
                                    ),
                        )
    # Customizing the figure   
    fig.update_layout(xaxis=dict(range=xlim), yaxis=dict(range=ylim))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    fig.update_layout(width=width, height=height)
    fig.update_layout(template=template, plot_bgcolor=plot_bgcolor)
    fig.update_layout(legend_orientation=legend_orientation)
    # fig.update_layout(legend=dict(x=legend_x, y=legend_y))
    fig.update_traces(mode=mode, opacity=opacity)

    # x and y axes settings
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='Black',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='black'
            ),
            tickwidth=2,
            tickcolor='Black',
            ticklen=5,
            tickangle=0,
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=True,
            showticklabels=True,
            linecolor='Black',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='black'
            ),
            tickwidth=2,
            tickcolor='Black',
            ticklen=5,
            tickangle=0,
        ),
    
        showlegend=False, #True,
        # legend=dict(
        #     x=legend_x,
        #     y=legend_y,
        #     traceorder="normal",
        #     font=dict(
        #         family="sans-serif",
        #         size=12,
        #         color="black"
        #     ),
        #     bgcolor="White",
        #     bordercolor="White",
        #     borderwidth=2
        # ),
            
    )

    # Saving the figure
    if save_dir != None:
        save_image(fig, save_dir, save_name, save_format, width, height)
    if show:
        fig.show()

def HeatMap(x, # 1D array of x-axis values
            y, # 1D array of values to be represented as colors
            z, # 2D array of values to be represented as colors
            title=None,
            x_label=None,
            y_label=None,
            z_label=None,
            colorscale = 'Jet', # color scale of the figure (Viridis, Cividis, Blackbody, Bluered, Blues, Earth, Electric, Greens, Greys, Hot, Jet, Picnic, Portland, Rainbow, RdBu, Reds, Viridis, YlGnBu, YlOrRd)
            reversescale=False, # reverse the color scale
            zmin=None, # min value of the color scale
            zmax=None, # max value of the color scale
            width = 600, height = 400, # width and height of the figure
            tickvals = None, # tick values
            ticktext = None, # tick text
            template='plotly_white', # template of the figure background (plotly, plotly_white, plotly_dark, ggplot2, seaborn, simple_white, none)
            plot_bgcolor='white', # background color of the figure
            save_dir=None, # directory to save the figure
            save_name='HeatMap', # name of the figure
            save_format=['png'], # format of the figure
            show=True, # show the figure or not
            ):
    '''
    HeatMap plot
    '''
    # Creating the figure
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=z, x=x, y=y,
                            colorscale=colorscale,
                            reversescale=reversescale,
                            zmin=zmin,
                            zmax=zmax,
                            colorbar=dict(
                                    title=z_label,
                                    titleside="right",
                                    # tickmode="array",
                                    # tickvals=tickvals,
                                    # ticktext=ticktext,
                                    # ticks="outside",
                                    ticklen=5,
                                    showticklabels=True,
                                    thickness=25,
                                    len=0.8,

                                    titlefont=dict(
                                            size=14,
                                            family='Arial, sans-serif'
                                    ),
                                    tickfont=dict(
                                            size=12,
                                            family='Arial, sans-serif'
                                    ),


                                            )))
    # Customizing the figure
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    fig.update_layout(xaxis=dict(range=[x.min().item(),x.max().item()]), yaxis=dict(range=[y.min().item(),y.max().item()]))
    fig.update_layout(width=width, height=height)
    fig.update_layout(template=template, plot_bgcolor=plot_bgcolor)
    if show:
        fig.show()
    # Saving the figure
    if save_dir != None:
        save_image(fig, save_dir, save_name, save_format, width, height)

def Bar(x, # 1D array of x-axis values
        y, # 1D array of y-axis values
        names=None, # names of the traces
        title=None,
        x_label=None,
        y_label=None,
        width=800, # width of the figure
        height=800, # height of the figure
        template='plotly_white', # template of the figure background (plotly, plotly_white, plotly_dark, ggplot2, seaborn, simple_white, none)
        plot_bgcolor='white', # background color of the figure
        save_dir=None, # directory to save the figure
        save_name='Bar', # name of the figure
        save_format=['png'], # format of the figure
        show=True, # show the figure or not
        ):
    '''
    Bar plot
    '''
    # Creating the figure
    fig = go.Figure()
    if names == None:
        fig.add_trace(go.Bar(x=x, y=y))
    else:
        for i in range(len(y)):
            fig.add_trace(go.Bar(x=x, y=y[i], name=names[i]))
    # Customizing the figure
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    fig.update_layout(width=width, height=height)
    fig.update_layout(template=template, plot_bgcolor=plot_bgcolor)
    # Saving the figure
    if save_dir != None:
        save_image(fig, save_dir, save_name, save_format, width, height)
    if show:
        fig.show()

def get_y_lim(y, index=0, y_ranges_scale=[[1, 1]], y_ranges_round=[[0, 0]]): 
    y_scale = y_ranges_scale[index]
    y_min, y_max = torch.min(y).item(), torch.max(y).item()
    y_range = y_max - y_min
    y_lim   = [y_min - y_scale[0]*y_range, y_max + y_scale[1]*y_range]
    y_lim   = [round(y_lim[0], y_ranges_round[index][0]), round(y_lim[1], y_ranges_round[index][1])]
    if y_lim[0] == y_lim[1]:
        y_lim[0] = y_lim[0] - 1
        y_lim[1] = y_lim[1] + 1
    return y_lim

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
# plotly.offline.init_notebook_mode(connected=True)

def get_data_ploty(plot_data, down_size=100):
    x = plot_data['x'].cpu().numpy()[::down_size]
    y = plot_data['y'].cpu().numpy() # we don't downsample the y values
    z = plot_data['z'].cpu().numpy()[:,::down_size]
    print('x shape:', x.shape, 'y shape:', y.shape, 'z shape:', z.shape)
    return x, y, z



def plot_data_ploty(data, config, parameters_analysis, save_name,  down_size=100, plot_heatmap = True, plot_analysis = True, labels=None, before_after_label=["", ""]):

    if plot_heatmap:
        # Create the first figure for the heatmap
        fig1 = make_subplots(rows=1, cols=2)
        heatmap = [None, None]

        for i, key in enumerate(['Reflectance_plot0', 'Reflectance_plot1']):
            plot_data = data[key]
            x, y, z = get_data_ploty(plot_data, down_size=down_size)
            colorscale = 'inferno' # 'hot', 'viridis' , 'inferno' , 'plasma' , 'magma' , 'cividis' , 'twilight' , 'twilight_shifted' , 'jet' , 'turbo' , 'thermal' , 'hsv' , 'gray' , 'bone' , 'pink' , 'spring' , 'summer' , 'autumn' , 'winter' , 'cool' , 'Wistia' , 'hot' , 'afmhot' , 'gist_heat' , 'copper'

            heatmap[i] = go.Heatmap(
                x=x,
                y=y,
                z=z,
                colorscale=colorscale
            )

        fig1.add_trace(heatmap[0], row=1, col=1)
        fig1.add_trace(heatmap[1], row=1, col=2)

        fig1.update_layout(
            xaxis_title=plot_data['x_label'],
            yaxis_title=plot_data['y_label'],
            coloraxis_colorbar=dict(title=plot_data['z_label'])
        )
        fig1.write_html(os.path.join(save_name + '_heatmap.html'))
        
        
    if plot_analysis:
        # Create the second figure with subplots
        # All keys for defect : ['band_gap_begining_plot0', 'band_gap_begining_plot1', 'band_gap_begining_data', 'band_gap_end_plot0', 'band_gap_end_plot1', 'band_gap_end_data', 'band_gap_width_plot0', 'band_gap_width_plot1', 'band_gap_width_data', 'band_gap_center_plot0', 'band_gap_center_plot1', 'band_gap_center_data', 'average_peak_intensity_plot0', 'average_peak_intensity_plot1', 'average_peak_intensity_data', 'peak_wavelength_plot0', 'peak_wavelength_plot1', 'peak_wavelength_data', 'position_left_defect_plot0', 'position_left_defect_plot1', 'position_left_defect_data', 'position_right_defect_plot0', 'position_right_defect_plot1', 'position_right_defect_data', 'FullWidthHalfMax_plot0', 'FullWidthHalfMax_plot1', 'FullWidthHalfMax_data', 'defect_intensity_T_plot0', 'defect_intensity_T_plot1', 'defect_intensity_T_data', 'defect_intensity_R_plot0', 'defect_intensity_R_plot1', 'defect_intensity_R_data', 'defect_peak_width_plot0', 'defect_peak_width_plot1', 'defect_peak_width_data', 'QualityFactor_plot0', 'QualityFactor_plot1', 'QualityFactor_data', 'Sensitivity_plot0', 'Sensitivity_data', 'FigureOfMerit_plot0', 'FigureOfMerit_data']
        # All keys for periodic : dict_keys(['band_gap_begining_plot0', 'band_gap_begining_plot1', 'band_gap_begining_data', 'band_gap_end_plot0', 'band_gap_end_plot1', 'band_gap_end_data', 'band_gap_width_plot0', 'band_gap_width_plot1', 'band_gap_width_data', 'band_gap_center_plot0', 'band_gap_center_plot1', 'band_gap_center_data', 'average_peak_intensity_plot0', 'average_peak_intensity_plot1', 'average_peak_intensity_data', 'peak_wavelength_plot0', 'peak_wavelength_plot1', 'peak_wavelength_data', 'position_left_defect_plot0', 'position_left_defect_plot1', 'position_left_defect_data', 'position_right_defect_plot0', 'position_right_defect_plot1', 'position_right_defect_data', 'FullWidthHalfMax_plot0', 'FullWidthHalfMax_plot1', 'FullWidthHalfMax_data', 'defect_intensity_T_plot0', 'defect_intensity_T_plot1', 'defect_intensity_T_data', 'defect_intensity_R_plot0', 'defect_intensity_R_plot1', 'defect_intensity_R_data', 'defect_peak_width_plot0', 'defect_peak_width_plot1', 'defect_peak_width_data', 'QualityFactor_plot0', 'QualityFactor_plot1', 'QualityFactor_data', 'Sensitivity_plot0', 'Sensitivity_data', 'FigureOfMerit_plot0', 'FigureOfMerit_data', 'Reflectance_plot0', 'Reflectance_plot1', 'Reflectance_data'])

        if parameters_analysis['strcuture_type'] == 'periodic_with_defect':
            keys = ['peak_wavelength_plot0', 'FullWidthHalfMax_plot0', 'QualityFactor_plot0', 'Sensitivity_plot0', 'FigureOfMerit_plot0', 'band_gap_begining_plot0', 'band_gap_end_plot0', 'band_gap_width_plot0',   ]
            keys2 = ['peak_wavelength_plot1', 'FullWidthHalfMax_plot1', 'QualityFactor_plot1', 'Sensitivity_plot0', 'FigureOfMerit_plot0', 'band_gap_begining_plot1', 'band_gap_end_plot1', 'band_gap_width_plot1',  ]
            rows, cols = 3, 3
        elif parameters_analysis['strcuture_type'] == 'periodic':
            keys  = ['band_gap_begining_plot0', 'band_gap_end_plot0', 'band_gap_width_plot0', 'Sensitivity_plot0', 'average_peak_intensity_plot0', 'band_gap_center_plot0',   ]
            keys2 = ['band_gap_begining_plot1', 'band_gap_end_plot1', 'band_gap_width_plot1', 'Sensitivity_plot0', 'average_peak_intensity_plot1', 'band_gap_center_plot0',  ]
            rows, cols = 2, 3
        else:
            raise ValueError('strcuture_type must be periodic_with_defect or periodic')
        
        
        texts = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)', '(o)', '(p)', '(q)', '(r)', '(s)', '(t)', '(u)', '(v)', '(w)', '(x)', '(y)', '(z)']
        colors = ['blue', 'red', 'green', 'black', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'lime', 'teal', 'coral', 'lightblue', 'lightgreen', 'lightcoral', 'lightpink', 'lightgray', 'lightolive', 'lightcyan', 'lightmagenta', 'lightyellow', 'lightlime', 'lightteal', 'lightcoral']

        fig2 = make_subplots(rows=rows, cols=cols, subplot_titles=texts)

        for i, key in enumerate(keys):
            plot_data = data[key]
            plot_data2 = data[keys2[i]]
            row, col = i // 3 + 1, i % 3 + 1

            x = plot_data['x'].cpu().numpy()
            y = plot_data['y'].cpu().numpy()

            x2 = plot_data2['x'].cpu().numpy()
            y2 = plot_data2['y'].cpu().numpy()

            trace = go.Scatter(x=x, y=y, mode='lines', line=dict(color=colors[i]),
                                name=(before_after_label[0]+str(labels[0])+before_after_label[1]) if labels is not None else None)
            trace2 = go.Scatter(x=x2, y=y2, mode='lines', line=dict(color=colors[i], dash='dash'),
                                
                                 name=(before_after_label[0]+str(labels[1])+before_after_label[1]) if labels is not None else None)

            fig2.add_trace(trace, row=row, col=col)
            fig2.add_trace(trace2, row=row, col=col)

            fig2.update_xaxes(title_text=plot_data['x_label'], row=row, col=col)
            fig2.update_yaxes(title_text=plot_data['y_label'], row=row, col=col)
            

        fig2.update_layout(height=900, width=800, showlegend=False)
        fig2.write_html(os.path.join(save_name + '_analysis.html'))

    return data

class visualization_config():
    @staticmethod
    def create_config(name):
        return {'scale':None, 'x_label':None, 'y_label':None, 'xlim':None, 'ylim':None,'add_to_word':visualization_config.add_to_word(name), 'del_points_x':None, 'del_points_y':None,}
    
    @staticmethod
    def add_to_word(name):
        add_to_word_dict = {'Reflectance1':0,
                            'peak_wavelength1':1,
                            'FullWidthHalfMax1':2,
                            'QualityFactor1':3,
                            'Sensitivity0':4,
                            'FigureOfMerit0':5
                            }
        if name in add_to_word_dict.keys():
            return add_to_word_dict[name]

class visualization():

    @staticmethod
    def visualize(path, quantity_dict, parameters_analysis, plot_dict={'all_in_one_plot':True, 'down_size':100, 'plot_heatmap':True, 'plot_analysis':True, 'labels':None, 'before_after_label':["", ""]}):
        # plot_dict = {'all_in_one_plot':True, 'down_size':100, 'plot_heatmap':True, 'plot_analysis':True, 'labels':None, 'before_after_label':["", ""]}

        for key_quantity_dict, _ in quantity_dict.items():

            save_name = key_quantity_dict 
            data_path, config_path = get_data_and_config_paths(path, save_name)
            data, config = load_data_and_config(data_path, config_path)
            path_plot = os.path.join(path, 'plot')
            os.makedirs(path_plot, exist_ok=True)
            save_name_plot = os.path.join(path_plot, save_name)
            if plot_dict['all_in_one_plot']:
                _ = plot_data_ploty(data, config, parameters_analysis, save_name_plot, down_size=plot_dict['down_size'], plot_heatmap=plot_dict['plot_heatmap'], plot_analysis=plot_dict['plot_analysis'], labels=plot_dict['labels'], before_after_label=plot_dict['before_after_label'])
            else:
                keys_to_remove = ['add_to_word']
                keys_to_add    = {'show':False,}
                remove_from_dicts(config, keys_to_remove)
                add_to_dicts(config, keys_to_add)

                for key, value in data.items():
                    value["save_name"] =  save_name + '_' + key
                    if "z" in value:
                        plot_type = 'HeatMap'
                    else:
                        plot_type = "plot"
                
                    visualization.plot_from_dict(plot_type, value, config[key], path)

    
    @staticmethod
    def plot_from_dict(plot_type, dict_data, over_write_dict, save_path):
        #TODO: Move this to visualize function
        # replace the parameters from over_write 
        values_to_remove = ['scale']
        for key, value in over_write_dict.items():
            if value is not None:
                dict_data[key] = value
            if key in values_to_remove and key in dict_data.keys():
                dict_data.pop(key)
        dict_data['save_dir'] = save_path

        # plot the data
        if plot_type == 'plot':
            plot(**dict_data)
        elif plot_type == 'HeatMap':
            [dict_data['x'], dict_data['z']] = pool_data([dict_data['x'], dict_data['z']], pool_size=4)
            HeatMap(**dict_data)
        elif plot_type == 'Bar':
            Bar(**dict_data)
        else:
            raise ValueError('plot_type must be plot, HeatMap or Bar')
