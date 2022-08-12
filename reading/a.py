import pickle

class Singer(object):
    def __init__(self, lylics):
        self.lylics = lylics
    def sing(self):
        print(self.lylics)

# singer = Singer('Shanranran')

with open('singer.pickle', 'wb') as f:
    pickle.dump(singer, f)

singer.sing()