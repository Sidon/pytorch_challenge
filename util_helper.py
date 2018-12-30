import random

def rnd_classes(batch, batch_classes,  dict_classes, k=5):
    batch_classes = list(batch_classes.numpy())
    choices = random.choices(range(len(batch)), k=k)
    imgs = [batch[c] for c in choices]
    labels = [(str(batch_classes[c]), dict_classes[str(batch_classes[c])]) for c in choices]

    return imgs, labels



  # def get_vgg(self):
  #           vggs = {}
  #           print('vgg11')
  #           vggs['vgg11'] = models.vgg11(pretrained=True)
  #           vggs['vgg11_bn'] = models.vgg11_bn(pretrained=True)
  #           vggs['vgg13'] = models.vgg13(pretrained=True)
  #           vggs['vgg13_bn'] = models.vgg13_bn(pretrained=True)
  #           vggs['vgg16'] = models.vgg16(pretrained=True)
  #           vggs['vgg16_bn'] = models.vgg16_bn(pretrained=True)
  #           vggs['vgg19'] = models.vgg19(pretrained=True)
  #           vggs['vgg19_bn'] = models.vgg19_bn(pretrained=True)
  #           return vggs
  #
  #   def get_densenet(self):
  #       ds169 = models.densenet169(pretrained=True)
  #       ds201 = models.densenet201(pretrained=True)
  #       print(ds169.classifier, ds201.classifier, sep='\n')
