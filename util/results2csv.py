import csv
import numpy as np
import os


def init_csv(path, name, fields=None):
    if fields is None:
        fields = ['Epoch', 'Iteration', 'D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
    csv_path = os.path.join(path, name)

    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    csv_path = os.path.join(csv_path, 'loss.csv')

    with open(csv_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(fields)


def append_loss(path, name, epoch, epoch_iter, loss_dict):
    csv_path = os.path.join(path, name)
    csv_path += '/loss.csv'
    row = [epoch, epoch_iter]
    for key in loss_dict.keys():
        row += [loss_dict[key]]

    with open(csv_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)


if __name__ == "__main__":
    res_path = "D:\\Development\\stain_normalization_CycleGAN\\results"
    name = "gray_norm_2"


    init_csv(res_path, name, header)
