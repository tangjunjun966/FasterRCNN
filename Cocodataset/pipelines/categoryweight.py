# 为特定类增加权重方法

class CategoryWeight():
    def __init__(self, cat_weight=None):

        self.cat_weight = cat_weight

    def __call__(self, results):
        cat_weight = dict()
        if results['mode'] in ['train', 'val'] and self.cat_weight is not None:
            assert isinstance(cat_weight, dict), 'cat_weight must be  dict'
            cat2class = results['cat2class']
            for k, v in cat2class.items():
                if v in self.cat_weight.keys():
                    cat_weight[k] = self.cat_weight[v]
                else:
                    cat_weight[k] = 1.0

            results['cat_weight']=cat_weight
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(cat_weight_val={})'.format(self.cat_weight)
        return repr_str
