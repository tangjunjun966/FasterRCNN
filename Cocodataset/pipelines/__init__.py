from .loadfromimgfile import LoadImageFromFile
from .loadannolations import LoadAnnotations
from .resize import Resize
from .randomflip import RandomFlip
from .randomcrop import RandomCrop
from .randomcropresize import RandomCropResize



from .pad import Pad
from .normalize import Normalize
from .categoryweight import CategoryWeight
from .defaultformatbundle import DefaultFormatBundle
from .collect import Collect


pipelines_cls = {'LoadImageFromFile': LoadImageFromFile,
                 'LoadAnnotations': LoadAnnotations,
                 'Resize': Resize,
                 'RandomFlip': RandomFlip,
                 'RandomCrop': RandomCrop,
                 'RandomCropResize': RandomCropResize,

                  'Pad': Pad,
                 'Normalize': Normalize,
                 'CategoryWeight': CategoryWeight,
                 'DefaultFormatBundle': DefaultFormatBundle,
                 'Collect': Collect

                 }
