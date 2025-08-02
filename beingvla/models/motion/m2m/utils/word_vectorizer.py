import numpy as np
import pickle as pkl
from os.path import join as pjoin

POS_enumerator = {
    'VERB': 0, 'NOUN': 1, 'DET': 2, 'ADP': 3, 'NUM': 4,
    'AUX': 5, 'PRON': 6, 'ADJ': 7, 'ADV': 8, 'Loc_VIP': 9,
    'Body_VIP': 10, 'Obj_VIP': 11, 'Act_VIP': 12, 'Desc_VIP': 13,
    'OTHER': 14
}


VIP_dict = {
    'Loc_VIP': ('left', 'right', 'clockwise', 'counterclockwise', 'anticlockwise',
               'forward', 'back', 'backward', 'up', 'down', 'straight', 'curve'),
    'Body_VIP': ('arm', 'chin', 'foot', 'feet', 'face', 'hand', 'mouth', 'leg',
                'waist', 'eye', 'knee', 'shoulder', 'thigh'),
    'Obj_VIP': ('stair', 'dumbbell', 'chair', 'window', 'floor', 'car', 'ball',
               'handrail', 'baseball', 'basketball'),
    'Act_VIP': ('walk', 'run', 'swing', 'pick', 'bring', 'kick', 'put', 'squat', 'throw', 'hop', 'dance', 'jump', 'turn',
               'stumble', 'dance', 'stop', 'sit', 'lift', 'lower', 'raise', 'wash', 'stand', 'kneel', 'stroll',
               'rub', 'bend', 'balance', 'flap', 'jog', 'shuffle', 'lean', 'rotate', 'spin', 'spread', 'climb'),
    'Desc_VIP': ('slowly', 'carefully', 'fast', 'careful', 'slow', 'quickly', 'happy', 'angry', 'sad', 'happily',
                'angrily', 'sadly')
}


class WordVectorizer:
    def __init__(self, meta_root, prefix):
        """Initialize word vectorizer with pre-trained embeddings."""
        vectors = np.load(pjoin(meta_root, f'{prefix}_data.npy'))
        words = pkl.load(open(pjoin(meta_root, f'{prefix}_words.pkl'), 'rb'))
        self.word2idx = pkl.load(open(pjoin(meta_root, f'{prefix}_idx.pkl'), 'rb'))
        self.word2vec = {w: vectors[self.word2idx[w]] for w in words}
    
    def _get_pos_onehot(self, pos):
        """Create one-hot vector for part-of-speech tag."""
        vec = np.zeros(len(POS_enumerator))
        vec[POS_enumerator.get(pos, POS_enumerator['OTHER'])] = 1
        return vec

    def __len__(self):
        return len(self.word2vec)

    def _get_vip_category(self, word):
        """Check if word belongs to any VIP category."""
        for category, words in VIP_dict.items():
            if word in words:
                return category
        return None
    
    def __getitem__(self, item):
        """
        Get word vector and POS vector for input item.
        Format: 'word/pos' or just 'word' (defaults to 'OTHER' POS)
        """
        parts = item.split('/')
        word = parts[0]
        pos = parts[1] if len(parts) > 1 else 'OTHER'
        
        # Handle unknown words
        if word not in self.word2vec:
            return self.word2vec['unk'], self._get_pos_onehot('OTHER')

        # Check for VIP category
        vip_category = self._get_vip_category(word)
        pos_vec = self._get_pos_onehot(vip_category if vip_category else pos)
        
        return self.word2vec[word], pos_vec
    

class WordVectorizerV2(WordVectorizer):
    def __init__(self, meta_root, prefix):
        """Extended version that adds word index to returned tuple."""
        super().__init__(meta_root, prefix)
        self.idx2word = {idx: w for w, idx in self.word2idx.items()}

    def __getitem__(self, item):
        word_vec, pos_vec = super().__getitem__(item)
        word = item.split('/')[0]
        word_idx = self.word2idx.get(word, self.word2idx['unk'])
        return word_vec, pos_vec, word_idx

    def itos(self, idx):
        """Convert index to string, with special handling for padding."""
        return "pad" if idx == len(self.idx2word) else self.idx2word.get(idx, "unk")
 
