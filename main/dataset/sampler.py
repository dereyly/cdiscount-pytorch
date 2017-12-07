from common import *
# common tool for dataset

#sampler -----------------------------------------------

class FixedSampler(Sampler):
    def __init__(self, data, list):
        self.num_samples = len(list)
        self.list = list

    def __iter__(self):
        #print ('\tcalling Sampler:__iter__')
        return iter(self.list)

    def __len__(self):
        #print ('\tcalling Sampler:__len__')
        return self.num_samples


# see trorch/utils/data/sampler.py
class RandomSampler1(Sampler):
    def __init__(self, data, num_samples=None):
        self.len_data = len(data)
        self.num_samples = num_samples or self.len_data

    def __iter__(self):
        #print ('\tcalling Sampler:__iter__')

        l=[]
        while 1:
            ll = list(range(self.len_data))
            random.shuffle(ll)
            l = l + ll
            if len(l)>=self.num_samples: break

        l= l[:self.num_samples]
        return iter(l)


    def __len__(self):
        #print ('\tcalling Sampler:__len__')
        return self.num_samples
