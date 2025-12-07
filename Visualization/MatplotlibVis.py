import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import torch

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
    
    plot_path = os.path.join(save_dir, f'{idx}sample_{idx}.png')
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
