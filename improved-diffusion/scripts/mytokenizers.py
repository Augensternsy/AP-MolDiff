# Licensed base: Diffusion-LM 759889d58ef38e2eed41a8c34db8032e072826f4 (Apache-2.0).
# OpenAI-derived portions retain their upstream MIT notices; see ../THIRD_PARTY_NOTICES.md.
# AP-MolDiff adapters: newly expressed from the frozen interface/behavior contract.
# Wsy-authorized local contributions remain subject to their MIT authorization.
# No assertion of full historical-model or paper reproduction is made.
import random
import torch

class _SmilesLexer:
    """Greedy SMILES lexical rules; unmatched characters are skipped as before."""
    def findall(self, text):
        result = []
        offset = 0
        single = set('BCNOSPFIbcnosp().=#-+\\/:~@?>*$0123456789')
        while offset < len(text):
            width = 0
            if text[offset] == '[':
                end = text.find(']', offset + 1)
                if end > offset + 1:
                    width = end + 1 - offset
            if not width and text[offset:offset+2] in ('Br', 'Cl'):
                width = 2
            if not width and text[offset] == '%' and len(text[offset+1:offset+3]) == 2:
                if all(c in '0123456789' for c in text[offset+1:offset+3]):
                    width = 3
            if not width and text[offset] in single:
                width = 1
            if width:
                result.append(text[offset:offset+width])
            offset += width or 1
        return result

class ChemformerTokenizer:
    def __init__(self, max_len=170):
        # Frozen ID contract: these entries are data, not a new vocabulary design.
        entries = ['[PAD]', '[SOS]', '[EOS]', '[UNK]', 'O', '=', 'C', '1', 'N', '(', '/', 'c',
                   '2', 'n', '[nH]', ')', '[C@H]', '[C@@H]', '3', 'Br', 'F', 'S', 'Cl',
                   '\\', '[N+]', '[O-]', '#', '4', 's', '-', 'o', '5', '[C@@]', '[C@]', '6', 'P', '7']
        self.idtotok = dict(enumerate(entries))
        self.toktoid = dict(zip(entries, range(len(entries))))
        self.vocab_size = self.len = len(entries)
        self.max_len = max_len
        self.rg = _SmilesLexer()

    def __len__(self):
        return self.vocab_size

    def _pack(self, tokens):
        ids = [1, *(self.toktoid.get(token, 3) for token in tokens if token is not None), 2]
        if len(ids) < self.max_len:
            ids.extend([0] * (self.max_len - len(ids)))
        else:
            ids = ids[:self.max_len]
            ids[-1] = 2
        return torch.tensor([ids], dtype=torch.long)

    def encode_one(self, smi):
        return self._pack(self.rg.findall(smi))

    def __call__(self, smis):
        inputs = [smis] if type(smis) is str else smis
        return torch.cat(tuple(self.encode_one(value) for value in inputs), dim=0)

    def decode_one(self, iter):
        return ''.join(self.idtotok[int(token.item())] for token in iter)

    def decode(self, ids):
        rows = [ids] if ids.ndim == 1 else ids
        return [self.decode_one(row) for row in rows]

    @staticmethod
    def _event():
        draw = random.random()
        return ((True, True) if draw < .3 else (True, False) if draw < .65 else (False, True))

    def corrupt_one(self, smi):
        tokens = self.rg.findall(smi)
        length = len(tokens) + 2
        if length > self.max_len:
            return self.encode_one(smi)
        parentheses, rings = self._event()
        positions = [[i for i, token in enumerate(tokens) if token in ('(', ')')],
                     [i for i, token in enumerate(tokens) if token.isnumeric()]]
        largest = max([1] + [int(tokens[i]) for i in positions[1]])
        enabled = [parentheses, rings]
        add = []
        for category in range(2):
            remove, insert = self._event()
            add.append(insert)
            if enabled[category] and positions[category] and remove:
                weights = ([.6, .2, .1, .1] if category == 0 else [.7, .2, .05, .05])
                count = random.choices([1, 2, 3, 4], weights=weights, k=1)[0]
                removed = set(random.choices(positions[category], weights=None, k=count))
                length -= len(removed)
                for location in removed:
                    tokens[location] = None
        for category in range(2):
            if enabled[category] and add[category]:
                count = random.choices([1, 2, 3], weights=[.8, .2, .1], k=1)[0]
                count = min(self.max_len - length, count)
                for _ in range(count):
                    location = random.randrange(len(tokens) + 1)
                    if category == 0:
                        token = '(' if random.random() < .5 else ')'
                    else:
                        token = str(random.randrange(1, largest + 1))
                    tokens.insert(location, token)
                    length += 1
        return self._pack(tokens)

    def corrupt(self, smis):
        inputs = [smis] if type(smis) is str else smis
        return torch.cat(tuple(self.corrupt_one(value) for value in inputs), dim=0)
