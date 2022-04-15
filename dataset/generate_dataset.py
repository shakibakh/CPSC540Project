from CPSC540Project.dataset import graph
import torchvision.models as models
from torchvision.models import *
from dataset import graph

def adjust_net(net, large_input=False):
    """
    Adjusts the first layers of the network so that small images (32x32) can be processed.
    :param net: neural network
    :param large_input: True if the input images are large (224x224 or more).
    :return: the adjusted network
    """
    net.expected_input_sz = 224 if large_input else 32

    if large_input:
        return net

    def adjust_first_conv(conv1, ks=(3, 3), stride=1):
        assert conv1.in_channels == 3, conv1
        ks_org = conv1.weight.data.shape[2:]
        if ks_org[0] > ks[0] or ks_org[1] or ks[1]:
            # use the center of the filters
            offset = ((ks_org[0] - ks[0]) // 2, (ks_org[1] - ks[1]) // 2)
            offset1 = ((ks_org[0] - ks[0]) % 2, (ks_org[1] - ks[1]) % 2)
            conv1.weight.data = conv1.weight.data[:, :, offset[0]:-offset[0]-offset1[0], offset[1]:-offset[1]-offset1[1]]
            assert conv1.weight.data.shape[2:] == ks, (conv1.weight.data.shape, ks)
        conv1.kernel_size = ks
        conv1.padding = (ks[0] // 2, ks[1] // 2)
        conv1.stride = (stride, stride)

    if isinstance(net, ResNet):

        adjust_first_conv(net.conv1)
        assert hasattr(net, 'maxpool'), type(net)
        net.maxpool = nn.Identity()

    elif isinstance(net, DenseNet):

        adjust_first_conv(net.features[0])
        assert isinstance(net.features[3], nn.MaxPool2d), (net.features[3], type(net))
        net.features[3] = nn.Identity()

    elif isinstance(net, (MobileNetV2, MobileNetV3)):  # requires torchvision 0.9+

        def reduce_stride(m):
            if isinstance(m, nn.Conv2d):
                m.stride = 1

        for m in net.features[:5]:
            m.apply(reduce_stride)

    elif isinstance(net, VGG):

        for layer, mod in enumerate(net.features[:10]):
            if isinstance(mod, nn.MaxPool2d):
                net.features[layer] = nn.Identity()

    elif isinstance(net, AlexNet):

        net.features[0].stride = 1
        net.features[2] = nn.Identity()

    elif isinstance(net, MNASNet):

        net.layers[0].stride = 1

    elif isinstance(net, ShuffleNetV2):

        net.conv1.stride = 1
        net.maxpool = nn.Identity()

    elif isinstance(net, GoogLeNet):

        net.conv1.stride = 1
        net.maxpool1 = nn.Identity()

    else:
        print('WARNING: the network (%s) is not adapted for small inputs which may result in lower performance' % str(
            type(net)))

    return net

  
def test_model_graph():  
  model_simple = nn.Sequential(
            nn.Linear(32,20),
            nn.ReLU(),
            nn.Linear(20,100),
            nn.Linear(100,1)
          )
  model = adjust_net(model_simple).eval()  # load the model (you can try any model here)
  Graph(model).visualize(node_size=200)
  g = Graph(model)
  print(g.n_nodes, g.edges)
  
test_model_graph()
