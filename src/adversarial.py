import torch

def load_model(path):
    raise NotImplementedError



def iterative_fsgm(image, model, loss_fn, step_size, num_steps):
    adversarial_image = image
    for i in range(num_steps):
        prediction = model(adversarial_image)
        loss = loss_fn(prediction, adversarial_image)
        gradient = torch.autograd.grad(loss, adversarial_image)
        gradient_sign = torch.sign(gradient)

        adversarial_image = adversarial_image + step_size * gradient_sign

    return adversarial_image