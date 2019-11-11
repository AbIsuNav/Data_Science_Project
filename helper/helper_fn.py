import networks


def compute_val_loss(unified_net, val_loader, device):
    """
    Computes the validation loss and return its float number.
    :param unified_net: the trained net.
    :param val_loader: validation data loader
    :param device: -
    :return: the float number representing the validation loss
    """
    val_loss = 0

    for i_batch, batch in enumerate(val_loader):
        img_batch, label_batch = batch['image'].to(device), batch['label'].to(device)
        val_pred = unified_net(img_batch)

        batch_loss = networks.WCEL(val_pred, label_batch)
        val_loss += batch_loss.item()

    val_avg = val_loss / len(val_loader)  # taking the average over all images
    return round(val_avg, 3)  # round to three floating points
