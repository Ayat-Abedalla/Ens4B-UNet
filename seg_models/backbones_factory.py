import copy
import efficientnet.model as eff
from classification_models.models_factory import ModelsFactory


class BackbonesFactory(ModelsFactory):
    _default_feature_layers = {

        # List of layers to take features from backbone in the following order:
        # (x16, x8, x4, x2, x1) - `x4` mean that features has 4 times less spatial
        # resolution (Height x Width) than input image.

       
        # ResNets
        'resnet34': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
        'resnet50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),

        # ResNeXt
        'resnext50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
        'resnext101': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),

        # DenseNet
        'densenet169': (367, 139, 51, 4),

        # SE models
        'seresnext50': (1078, 584, 254, 4),
        'seresnext101': (2472, 584, 254, 4),

   
        # EfficientNets
      
        'efficientnetb3': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb4': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),

    }

    _models_update = {
        
        'efficientnetb3': [eff.EfficientNetB3, eff.preprocess_input],
        'efficientnetb4': [eff.EfficientNetB4, eff.preprocess_input],

    }

    @property
    def models(self):
        all_models = copy.copy(self._models)
        all_models.update(self._models_update)
        return all_models

    def get_backbone(self, name, *args, **kwargs):
        model_fn, _ = self.get(name)
        model = model_fn(*args, **kwargs)
        return model

    def get_feature_layers(self, name, n=5):
        return self._default_feature_layers[name][:n]

    def get_preprocessing(self, name):
        return self.get(name)[1]


Backbones = BackbonesFactory()
