# Importing common libraries
import os                         
import numpy as np               
import torch
import plotly.graph_objects as go  


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
