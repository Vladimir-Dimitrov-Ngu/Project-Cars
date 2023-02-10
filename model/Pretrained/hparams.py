config = dict(
    batch_size=64,
    learning_rate=1e-5,
    weight_decay=0.01,
    epochs=10,
    zero_init_residual=False,
    num_classes = 196,
    name = 'resnet152 + mixed',
    precision = 'mixed', # ['full', 'half', 'mixed']
    loss_scaling = True,
    optimizer = 'SGD',
    time = True
)