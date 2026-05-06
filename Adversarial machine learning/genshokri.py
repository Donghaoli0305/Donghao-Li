
"""
    Taken from HW3
"""
import joblib
import os

import numpy as np
import keras
import torch

from sklearn.base import clone
import sklearn.metrics as metrics

from sklearn.svm import LinearSVC
import utils
from part1 import load_and_grab
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--- Device: {device} ---")
print("-------------------")


"""
## Compute attack performance metrics, i.e., accuracy and advantage (assumes baseline = 0.5)
## Note: also returns the full confusion matrix
"""
def attack_performance(in_or_out_test, in_or_out_pred):
    cm = metrics.confusion_matrix(
        in_or_out_test.astype(int), in_or_out_pred.astype(int))
    accuracy = np.trace(cm) / np.sum(cm.ravel())
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    advantage = tpr - fpr

    return accuracy, advantage, cm


"""
## Extract 'sz' targets from in/out data
"""
def get_targets(x_in, y_in, x_out, y_out, sz=5000):

    x_temp = np.vstack((x_in, x_out))
    y_temp = np.vstack((y_in, y_out))

    inv = np.ones((x_in.shape[0], 1))
    outv = np.zeros((x_out.shape[0], 1))
    in_out_temp = np.vstack((inv, outv))
    assert x_temp.shape[0] == y_temp.shape[0]

    if sz > x_temp.shape[0]:
        sz = x_temp.shape[0]

    perm = np.random.permutation(x_temp.shape[0])
    perm = perm[0:sz]
    x_targets = x_temp[perm, :]
    y_targets = y_temp[perm, :]

    in_out_targets = in_out_temp[perm, :]

    return x_targets, y_targets, in_out_targets



"""
## Train attack models using the 'shadow training' technique of Shokri et al.
## Inputs:
##  - x_aux, y_aux: auxiliary data
##  - target_train_size: size of training data of target model
##  - create_model_fn: function to create a model of the same type as the target model
##  - train_model_fn: function to train a model of the same type as the target model [invoke as: train_model_fn(model, x, y)]
##  - num_shadow: number of shadow models (default: 4)
##  - attack_model_fn: function to create an attack model with scikit-learn
##
##  Output:
##  - attack_models: list of attack models, one per class.
"""

def shokri_attack_models(x_aux, y_aux, target_train_size, create_model_fn, train_model_fn, predict_fn, num_shadow=4, attack_model_obj=None):
    assert attack_model_obj is not None, "Failed to specify an attack model object."
    print(f"Attack model: {attack_model_obj.__class__}")
    assert 2*target_train_size < x_aux.shape[0]

    num_classes = y_aux.shape[1]
    class_train_list = [None] * num_classes

    def add_to_list(data):
        for label in range(0, num_classes):
            dv = data[data[:, -2] == label, :]
            col_idx = [i for i in range(0, num_classes)]
            col_idx.append(num_classes+1)
            if class_train_list[label] is None:
                class_train_list[label] = dv[:, col_idx]
            else:
                class_train_list[label] = np.vstack(
                    (class_train_list[label], dv[:, col_idx]))

    def random_subdataset(x, y, sz):
        assert x.shape[0] == y.shape[0]
        perm = np.random.permutation(x.shape[0])
        perm = perm[0:sz]

        return x[perm, :].copy(), y[perm, :].copy()

    for i in range(0, num_shadow):
        ## TODO ##
        # Insert your code here to train the ith shadow model and obtain the corresponding training data for the attack model
        # You can use random_subdataset() to sample a subdataset from aux and add_to_list() to populate 'class_train_list'
        print(f'Training shadow model #{i}')

        # Get twice the target training size from auxiliary data. Half for in and half for out
        x, y = random_subdataset(x_aux, y_aux, target_train_size * 2)
        x_train = x[0:target_train_size, :]
        y_train = y[0:target_train_size, :]
        x_test = x[target_train_size:, :]
        y_test = y[target_train_size:, :]
        model = create_model_fn()
        train_model_fn(model, x_train, y_train)

        # Add probability vector, true label, in/out label to the attack model training data
        y_pred_train = predict_fn(model, x_train)
        y_true_train = np.argmax(y_train, axis=1).reshape(-1, 1)
        add_to_list(np.concatenate(
            [y_pred_train, y_true_train, np.ones(y_true_train.shape)], axis=1))

        y_pred_test = predict_fn(model, x_test)
        y_true_test = np.argmax(y_test, axis=1).reshape(-1, 1)
        add_to_list(np.concatenate(
            [y_pred_test, y_true_test, np.zeros(y_true_test.shape)], axis=1))

    print("Training attack models...")
    # now train the models
    attack_models = []

    for label in range(0, num_classes):
        data = class_train_list[label]
        np.random.shuffle(data)
        x_data = data[:, :-1]
        y_data = data[:, -1]

        # train attack model
        am = clone(attack_model_obj)
        am = am.fit(x_data, y_data)
        attack_models.append(am)

    return attack_models

"""
## Perform the Shokri et al. attack
## Inputs:
##  - attack_models: list of attack models, one per class.
##  - x_targets, y_targets: records to attack
##  - query_target_model: function to query the target model [invoke as: query_target_model(x)]

##  Output:
##  - in_or_out_pred: in/out prediction for each target
"""

def do_shokri_attack(attack_models, x_targets, y_targets, query_target_model):

    num_classes = y_targets.shape[1]
    assert len(attack_models) == num_classes
    y_targets_labels = np.argmax(y_targets, axis=-1)

    in_or_out_pred = np.zeros((x_targets.shape[0],))

    pv = query_target_model(x_targets)
    assert pv.shape[0] == y_targets_labels.shape[0]

    for i in range(0, pv.shape[0]):
        label = y_targets_labels[i]
        assert 0 <= label < num_classes

        am = attack_models[label]
        in_or_out_pred[i] = am.predict(pv[i, :].reshape(1, -1))

    return in_or_out_pred



def train_model(model, x_train, y_train, num_epochs, batch_size=64, verbose=False, device="cuda"):
    """
        Trains the torch model given 'x_train' and 'y_train'
        Generated by GitHub Copilot
    """
    # select device
    device = torch.device('cuda' if device ==
                          'cuda' and torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    # optimizer and loss
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # prepare tensors
    x_t = torch.from_numpy(x_train.astype(np.float32))
    # if channels-last (N,H,W,C) -> convert to (N,C,H,W)
    if x_t.ndim == 4 and x_t.shape[-1] in (1, 3):
        x_t = x_t.permute(0, 3, 1, 2)

    # convert one-hot labels to class indices if needed
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        y_t = torch.from_numpy(np.argmax(y_train, axis=1).astype(np.int64))
    else:
        y_t = torch.from_numpy(y_train.astype(np.int64).reshape(-1))

    dataset = torch.utils.data.TensorDataset(x_t, y_t)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            if outputs.ndim > 2:
                outputs = outputs.view(outputs.shape[0], -1)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        # print progress occasionally
        if verbose and ((epoch + 1) == 1 or (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs):
            print(
                f'epoch {epoch+1}/{num_epochs}  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}')

    model.eval()
    return model


@torch.no_grad()
def pv_predict(model, x, device="cuda"):
    x = torch.Tensor(x).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).to("cpu").numpy()
    return probs


# create the model object
model_fp = './target_model.pt'
assert os.path.exists(model_fp)  # model must exist

model, fg = utils.load_model(model_fp, device=device)
assert fg == "0CCE0F932C863D6648E0", f"Modified model file {model_fp}!"

# Use validation data as auxiliary data to train shadow models
val_loader = utils.make_loader(
    './data/valtest.npz', 'val_x', 'val_y', batch_size=512, shuffle=False)
utils.check_loader(val_loader)
x_aux, y_aux = utils.grab_from_loader(val_loader, 100)
x_aux = np.array(x_aux)
y_aux = np.array(y_aux)
y_aux = keras.utils.to_categorical(y_aux, num_classes=10)

# Arguments for shokri_attack_models()

TARGET_TRAIN_SIZE = 2000
NUM_EPOCHS = 15
NUM_SHADOW = 10

# This is the empty model used to load the target model
target_model_train_fn = lambda: utils.get_resnet18_cifar()
create_model_fn = target_model_train_fn

train_model_fn = lambda model, x, y: train_model(model, x, y, NUM_EPOCHS, device=device, verbose=True)
predict_fn = lambda model, x: pv_predict(model, x, device=device)
attack_model_fn = LinearSVC()
query_target_model = lambda x: pv_predict(model, x, device=device)

# Train the attack models
attack_models = shokri_attack_models(
    x_aux, y_aux, TARGET_TRAIN_SIZE, create_model_fn, train_model_fn, predict_fn, num_shadow=NUM_SHADOW, attack_model_obj=attack_model_fn)


# Arguments for do_shokri_attack()
in_x, in_y = load_and_grab('./data/members.npz', 'members', num_batches=100)
out_x, out_y = load_and_grab(
    './data/nonmembers.npz', 'nonmembers', num_batches=100)
in_y = keras.utils.to_categorical(in_y, num_classes=10)
out_y = keras.utils.to_categorical(out_y, num_classes=10)
x_targets, y_targets, in_or_out_targets = get_targets(
    in_x, in_y, out_x, out_y, sz=TARGET_TRAIN_SIZE*2)

in_or_out_pred = do_shokri_attack(
    attack_models, x_targets, y_targets, query_target_model)

accuracy, advantage, _ = attack_performance(in_or_out_targets, in_or_out_pred)
print(
    f"Shokri et al. attack accuracy: {accuracy:.4f}, advantage: {advantage:.4f}")

# Save attack models inside shokri_attack/
OUT_DIR = './shokri_attack'
os.makedirs(OUT_DIR, exist_ok=True)
for idx, am in enumerate(attack_models):
    model_fp = os.path.join(OUT_DIR, f'attack_model_class_{idx}.pkl')
    joblib.dump(am, model_fp)
    print(f'Saved attack model {idx} -> {model_fp}')


