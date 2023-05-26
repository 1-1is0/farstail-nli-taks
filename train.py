import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
first_loss = True
first_acc = True


def draw_loss_curve(current_epoch, net_name, optimizer_name, loss_name, res):
    global first_loss
    plt.clf()
    x_epoch = list(range(1, current_epoch+1))
    loss_train = res["loss_train"]
    loss_val = res["loss_val"]
    plt.plot(x_epoch, loss_train, 'bo-', label='train')
    plt.plot(x_epoch, loss_val, 'ro-', label='val')
    if first_loss:
        plt.legend()
        first_loss = False
    os.makedirs("loss_graphs", exist_ok=True)
    plt.savefig(os.path.join('./loss_graphs',
                f'train_loss_{net_name}-{optimizer_name}_{loss_name}.jpg'))


def draw_acc_curve(current_epoch, net_name, optimizer_name, loss_name, res):
    global first_acc
    plt.clf()
    x_epoch = list(range(1, current_epoch+1))
    acc_train = res["acc_train"]
    acc_val = res["acc_val"]
    plt.plot(x_epoch, acc_train, 'bo-', label='train')
    plt.plot(x_epoch, acc_val, 'ro-', label='val')
    if first_acc:
        plt.legend()
        first_acc = False
    os.makedirs("loss_graphs", exist_ok=True)
    plt.savefig(os.path.join('./loss_graphs',
                f'train_acc_{net_name}-{optimizer_name}_{loss_name}.jpg'))


def total_accuracy(net, dataset, data_loader, device):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in data_loader.test:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f'Accuracy of the network on the {len(dataset.test)} test images: {100 * correct // total} %')


def get_state_path(net, criterion, optimizer):
    path = "model"
    net_name = net._get_name()
    criterion_name = criterion.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    os.makedirs(path, exist_ok=True)
    state_file_name = f"{path}/state-{net_name}-optimizer-{optimizer_name}-loss-{criterion_name}.pth"
    return state_file_name


def get_model_wights():
    state_file_name = get_state_path()
    state = torch.load(state_file_name)
    return state["state_dict"]


def get_model_best_wights():
    state_file_name = get_state_path()
    state = torch.load(state_file_name)
    return state["best_net_state"]
# %%


def train(model, criterion, optimizer, data_loader, dataset, device, epochs=20):
    path = "model"
    net_name = model._get_name()
    criterion_name = criterion.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    os.makedirs(path, exist_ok=True)
    state_file_name = f"{path}/state-{net_name}-optimizer-{optimizer_name}-loss-{criterion_name}.pth"
    state_res = {}
    best_net_state = None

    print(state_file_name, end=" ")
    if os.path.exists(state_file_name):
        print("exist")
        state = torch.load(state_file_name)
        model.load_state_dict(state["state_dict"])
        best_net_state = state.get("best_net_state", None)
        optimizer.load_state_dict(state["optimizer"])
        state_res = state["res"]
    else:
        print("Not exist")
    res = {
        "loss_train": state_res.get("loss_train", []),
        "loss_val": state_res.get("loss_val", []),
        "acc_val": state_res.get("acc_val", []),
        "acc_train": state_res.get("acc_train", []),
        "epoch": state_res.get("epoch", 0),
    }
    best_val_loss = min(res["loss_val"] + [9999])

    # loop over the dataset multiple times
    try:
        saving = False
        for epoch in range(res["epoch"]+1, epochs+1):
            start_time = time.time()
            running_loss = 0.0
            phase_loss = 0
            for phase in ["train", "val"]:
                total, correct = (0, 0)
                if phase == "train":
                    model.train(True)  # Set model to training mode
                else:
                    model.train(False)  # Set model to evaluate mode

                loop_iter = tqdm(enumerate(data_loader[phase], 0), total=len(
                    data_loader[phase]), leave=False)
                for i, data in loop_iter:
                    # get the inputs; data is a list of [inputs, labels]

                    input_ids = data["input_ids"].to(device)
                    attention_mask = data["attention_mask"].to(device)
                    token_type_ids = data["token_type_ids"].to(device)
                    label_id = data["label_id"].to(device)
                    now_batch_size = label_id.size(0)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    logits, probs = model(input_ids, attention_mask, token_type_ids)
                    # outputs = outputs.reshape(-1)
                    loss = criterion(logits, label_id)
                    if phase == "train":
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                        optimizer.step()
                    label_pred = torch.argmax(probs, axis=1)
                    total += label_id.size(0)
                    correct += (label_pred == label_id).sum().item()
                    # print statistics
                    phase_loss += loss.item() * now_batch_size
                    running_loss += loss.item()
                    loop_iter.set_postfix(
                        {
                            "epoch": epoch,
                            "phase": phase,
                            "loss": f"{running_loss / (i+1):04.4f}"
                        })
                phase_loss = phase_loss / len(dataset[phase])
                if phase == "val" and phase_loss <= best_val_loss:
                    best_val_loss = phase_loss
                    best_net_state = model.state_dict()
                    print("### BETTER NET STATE ###")
                # y_loss[phase].append(phase_loss)

                res[f"acc_{phase}"].append(round(100*correct/total, 2))
                res[f"loss_{phase}"].append(phase_loss)
                res["epoch"] = epoch

            end_time = time.time() - start_time
            print(
                f"[{end_time:.0f}s] Epoch {epoch} loss : {res['loss_train'][-1]:.8f} acc: {res['acc_train'][-1]} val: {res['loss_val'][-1]:.8f} acc: {res['acc_val'][-1]}%")
            # if epoch % 5 == 0 or epoch in (1, epochs-1):
            #     total_accuracy()
            draw_loss_curve(epoch, net_name=net_name, optimizer_name=optimizer_name,
                            loss_name=criterion_name, res=res)

            draw_acc_curve(epoch, net_name=net_name, optimizer_name=optimizer_name,
                           loss_name=criterion_name, res=res)
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_net_state": best_net_state,
                "optimizer": optimizer.state_dict(),
                "res": res
            }
            saving = True
    except KeyboardInterrupt:
        print("Stopping", end=" ")
        if 'state' in locals() and "phase" in locals() and saving:  # check if variable exist
            print("Saving")
            torch.save(state, state_file_name)
            return state
        print("phase", phase, "saving", saving)
        print()
        return None

    torch.save(state, state_file_name)
    return state
