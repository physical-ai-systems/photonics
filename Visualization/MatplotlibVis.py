import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import torch
from Methods.TransferMatrixMethod.Structure import Structure
from Methods.TransferMatrixMethod.Layer import Layer

def plot_sample_spectrum(wavelengths, target_R, pred_R, idx, loss, save_dir):
    """
    Plots the target and predicted spectra using Matplotlib.
    """
    fig, ax = plt.subplots()
    
    ax.plot(wavelengths, target_R, label='Target')
    ax.plot(wavelengths, pred_R, label='Output', linestyle='--')
    ax.legend()
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance')
    ax.set_title(f'Sample {idx} Loss: {loss:.6f}')
    
    plot_path = os.path.join(save_dir, f'{idx}_sample_{idx}.png')
    plt.savefig(plot_path)
    plt.close(fig)

def plot_structure(structure, save_path=None, title="Structure Visualization"):
    """
    Visualizes the multi-layer structure.
    Plots rectangles representing layers with thickness and material colors.
    
    Args:
        structure: The Structure object containing layers.
        save_path: Path to save the plot. If None, shows the plot.
        title: Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    layers = structure.layers
    
    # Collect unique materials for coloring
    unique_materials = []
    for layer in layers:
        mat_name = layer.material.name
        if mat_name not in unique_materials:
            unique_materials.append(mat_name)
            
    # Generate colors
    # Use a colormap that has distinct colors
    cmap = plt.get_cmap('tab20') 
    colors = {name: cmap(i % 20) for i, name in enumerate(unique_materials)}
    
    current_position = 0.0
    max_height = 1.0
    
    for i, layer in enumerate(layers):
        thickness = layer.thickness
        if isinstance(thickness, torch.Tensor):
            thickness = thickness.item()
            
        # Handle case where thickness might be 0 or very small
        if thickness <= 0:
            continue
        
        mat_name = layer.material.name
        color = colors[mat_name]
        
        # Create rectangle
        rect = patches.Rectangle(
            (current_position, 0), 
            thickness, 
            max_height, 
            linewidth=1, 
            edgecolor='black', 
            facecolor=color
        )
        ax.add_patch(rect)
        
        # Add text annotation for material name and thickness
        label_text = f"{mat_name}\n{thickness:.1f}nm"
        ax.text(
            current_position + thickness / 2, 
            max_height / 2, 
            label_text, 
            ha='center', 
            va='center', 
            rotation=90, 
            fontsize=8,
            color='white' if sum(color[:3]) < 1.5 else 'black',
            clip_on=True
        )

        current_position += thickness

    ax.set_xlim(0, current_position)
    ax.set_ylim(0, max_height)
    ax.set_xlabel('Thickness (nm)')
    ax.set_yticks([])
    ax.set_title(title)
    
    # Create a custom legend
    handles = [patches.Patch(color=colors[name], label=name) for name in unique_materials]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

def save_structure_plots(dataset, material_choice, pred_thickness, target_thickness, save_dir, prefix, title_suffix):
    """
    Plots predicted and target structures.
    
    Args:
        dataset: The dataset object containing Materials.
        material_choice: Tensor of shape (num_layers,) containing material indices.
        pred_thickness: Tensor of shape (num_layers,) containing denormalized predicted thickness.
        target_thickness: Tensor of shape (num_layers,) containing denormalized target thickness.
        save_dir: Directory to save the plots.
        prefix: Prefix for the filename.
        title_suffix: Suffix for the plot title.
    """
    if not hasattr(dataset, 'Materials'):
        return

    material_names = list(dataset.Materials.keys())
    layers_vis_pred = []
    layers_vis_target = []
    
    num_layers = pred_thickness.shape[0]

    for l_idx in range(num_layers):
        # Ensure material choice is integer
        mat_idx = int(material_choice[l_idx].item())
        if mat_idx >= len(material_names):
            mat_idx = 0
        
        mat_name = material_names[mat_idx]
        material = dataset.Materials[mat_name]
        
        layers_vis_pred.append(Layer(material, thickness=pred_thickness[l_idx]))
        layers_vis_target.append(Layer(material, thickness=target_thickness[l_idx]))
        
    structure_vis_pred = Structure(layers=layers_vis_pred, mode='TE', layers_parameters={'method': 'multi_layer'})
    plot_structure(structure_vis_pred, save_path=os.path.join(save_dir, f'{prefix}_structure_pred.png'), title=f'Structure Pred {title_suffix}')
    
    structure_vis_target = Structure(layers=layers_vis_target, mode='TE', layers_parameters={'method': 'multi_layer'})
    plot_structure(structure_vis_target, save_path=os.path.join(save_dir, f'{prefix}_structure_target.png'), title=f'Structure Target {title_suffix}')


def plot_first_sample(tb_logger, dataset, batch, R_calc, epoch, output_for_plot):
    """
    Plots the first sample of the batch to TensorBoard log directory.
    """
    try:
        save_dir = os.path.join(tb_logger.log_dir, 'plots')
        os.makedirs(save_dir, exist_ok=True)
        
        wavelengths = dataset.wavelength.values
        if wavelengths.ndim > 1:
            wavelengths = wavelengths[0]
        wavelengths = wavelengths.cpu().numpy().squeeze()
        
        target_R = batch['R'][0].cpu().numpy()
        pred_R = R_calc[0].cpu().numpy()
        
        sample_loss = torch.nn.functional.mse_loss(R_calc[0], batch['R'][0].to(R_calc.device)).item()
        
        plot_sample_spectrum(wavelengths, target_R, pred_R, epoch + 1, sample_loss, save_dir)

        # Plot structure
        thickness_min = dataset.thickness_range[0]
        thickness_max = dataset.thickness_range[1]
        
        pred_thickness_denorm = output_for_plot[0] * (thickness_max - thickness_min) + thickness_min
        target_thickness_denorm = batch['layer_thickness'][0] * (thickness_max - thickness_min) + thickness_min
        
        save_structure_plots(dataset, batch['material_choice'][0], pred_thickness_denorm, target_thickness_denorm, save_dir, f'{epoch+1}', f'Epoch {epoch+1}')
    except Exception as e:
        print(f"Failed to plot first sample for epoch {epoch+1}: {e}")