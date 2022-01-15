import torch


def reptile_learner(model, queue, optimizer, iteration, args):
    model.train()

    # Store the current model parameters
    prev_params = [param.data.clone() for param in model.parameters()]

    queue_length = len(queue)
    losses = 0

    for k in range(args.update_step):
        for i in range(queue_length):
            optimizer.zero_grad()

            data = queue[i]["batch"][k]
            task = queue[i]["task"]

            output = model.forward(task, data)

            loss = output[0].mean()
            loss.backward()
            losses += loss.detach().item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

    # Update model parameters
    beta = args.beta * (1 - iteration / args.meta_iteration)
    for idx, param in enumerate(model.parameters()):
        param.data = (1 - beta) * prev_params[idx].data + beta * param.data

    return losses / (queue_length * args.update_step)
