import random

def rnd_classes(batch, batch_classes,  dict_classes, k=5):
    batch_classes = list(batch_classes.numpy())
    choices = random.choices(range(len(batch)), k=k)
    imgs = [batch[c] for c in choices]
    labels = [(str(batch_classes[c]), dict_classes[str(batch_classes[c])]) for c in choices]

    return imgs, labels


