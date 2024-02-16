def plot_images(images, labels, num_rows=4, num_cols=8, title='Images from Train Dataset', infer=False):
    """
    Plots a grid of images with their corresponding labels.

    Args:
    - images (list): List of images to be plotted.
    - labels (list): List of labels corresponding to the images.
    - num_rows (int): Number of rows in the grid layout (default is 4).
    - num_cols (int): Number of columns in the grid layout (default is 8).
    - title (str): Title of the plot (default is 'Images from Train Dataset').
    - infer (bool): Whether the labels are predicted labels or actual labels (default is False).

    Returns:
    - None

    """
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    fig.suptitle(title, fontsize=16)

    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            ax = axes[i, j]
            ax.imshow(np.squeeze(images[index]), cmap='gray')
            if infer:
                ax.set_title(f'Predicted: {labels[index]}')
            else:
                ax.set_title(f'Label: {labels[index].item()}')
            ax.axis('off')

    plt.show()

def infer(model, image, transform):
    """
    Infers the label of an image using a given model.

    Args:
    - model: Trained model used for inference.
    - image: Input image to be inferred.
    - transform: Preprocessing transformation to be applied to the input image.

    Returns:
    - predicted (int): Predicted label for the input image.

    """
    model.eval()
    if transform is not None:
        image = transform(image)
    image = image.unsqueeze(0)
    output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()
