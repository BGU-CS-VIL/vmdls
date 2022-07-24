from torchvision import transforms
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


data_transforms = {'images': {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])},
    'mnist': {
        'train':transforms.Compose([
            transforms.RandomRotation(10, fill=(0,)),
            transforms.RandomResizedCrop(28, scale=(0.95,1)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,), (0.5,))]),
        'val':transforms.Compose([            
            transforms.ToTensor(),
            transforms.Normalize(
               (0.5,), (0.5,))]),
        'test':transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,), (0.5,))])},
    'emnist': {
        'train':transforms.Compose([
            transforms.RandomPerspective(),
            transforms.RandomRotation(10, fill=(0,)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,))]),
        'val':transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,))]),
        'test':transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,))])},
    'cifar': {
        'train':transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        'val':transforms.Compose([
            transforms.ToTensor(),
           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        'test':transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])},
    'toy': {
        'train':transforms.Compose([
            transforms.ToTensor()]),
        'val':transforms.Compose([
            transforms.ToTensor()]),
        'test':transforms.Compose([
            transforms.ToTensor()])},
    'cifar_test': {
        'train':transforms.Compose([#transforms.Re(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        'val':transforms.Compose([#transforms.CenterCrop(32),
            transforms.ToTensor(),
           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        'test':transforms.Compose([#transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])},
    'toy': {
        'train':transforms.Compose([
            transforms.ToTensor()]),
        'val':transforms.Compose([
            transforms.ToTensor()]),
        'test':transforms.Compose([
            transforms.ToTensor()])},
    # 'blob':{
    #     'train':transforms.Compose(
    #             [transforms.ToTensor()]
    #         ),
    #     'val':transforms.Compose(
    #         [transforms.ToTensor()]
    #     ),
    #     'test':transforms.Compose(
    #         [transforms.ToTensor()]
    #     ),
    # }
    'blob':{
        'train':transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]
            ),
        'val':transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]
        ),
        'test':transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]
        ),
    }
}