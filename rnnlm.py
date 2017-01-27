import dynet
import random
import util
import math
import seq2seq


###########################################################################
class RNNLanguageModel(object):
    name = "template"


    def __init__(self, model, vocab, args):
        self.model = model
        self.vocab = vocab
        self.args = args


    def BuildLMGraph(self, sent, sent_args=None):
        pass


    def sample(self, first=0, stop=-1, nchars=100):
        pass


def get_lm(name):
    for c in util.itersubclasses(RNNLanguageModel):
        if c.name == name: return c
    raise Exception("no language model found with name: " + name)
##########################################################################


class BaselineRNNLM(RNNLanguageModel):
    name = "baseline"


    def __init__(self, model, vocab, args):
        self.m = model
        self.vocab = vocab
        self.args = args


        if args.s2s:
            print "loading s2s..."
            self.s2s = seq2seq.get_s2s(args.s2s_type).load(model, args.s2s)
        
        self.rnn = args.rnn(args.layers, args.input_dim, args.hidden_dim, model)


        self.lookup = model.add_lookup_parameters((vocab.size, args.input_dim))
        self.R = model.add_parameters((vocab.size, args.hidden_dim))
        self.bias = model.add_parameters((vocab.size,))


    def BuildLMGraph(self, sent, sent_args=None):
        dynet.renew_cg()
        init_state = self.rnn.initial_state()


        R = dynet.parameter(self.R)
        bias = dynet.parameter(self.bias)
        errs = [] # will hold expressions
        state = init_state


        for (cw,nw) in zip(sent,sent[1:]):
            x_t = self.lookup[cw]
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            err = dynet.pickneglogsoftmax(r_t, int(nw))
            errs.append(err)
        nerr = dynet.esum(errs)
        return nerr


    def BuildLMGraph_batch(self, sents, sent_args=None):
        dynet.renew_cg()
        init_state = self.rnn.initial_state()
        mb_size = len(sents)
        #MASK SENTENCES
        wids = [] # Dimension: maxSentLength * minibatch_size


        # List of lists to store whether an input is 
        # present(1)/absent(0) for an example at a time step
        masks = [] # Dimension: maxSentLength * minibatch_size


        #No of words processed in this batch
        tot_words = 0
        maxSentLength = max([len(sent) for sent in sents])
        sentLengths =[len(sent) for sent in sents]


        for k in range(maxSentLength):
            wids.append([(self.vocab.s2t[sent[k]].i if len(sent)>k else self.vocab.END_TOK.i) for sent in sents])
            mask = [(1 if len(sent)>k else 0) for sent in sents]
            masks.append(mask)
            tot_words += sum(mask)


        # print "WIDS:", wids


        R = dynet.parameter(self.R)
        bias = dynet.parameter(self.bias)
        losses = [] # will hold losses
        state = init_state


        for (mask, curr_words, next_words) in zip(masks,wids,wids[1:]):
            # print "Current words: ", curr_words
            # print "Next words: ", next_words
            x_t = dynet.lookup_batch(self.lookup, curr_words) 
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            loss = dynet.pickneglogsoftmax_batch(r_t, next_words)
            # loss is a list of losses
            # mask the loss if at least one sentence is shorter
            if 0 in mask:
                mask_expr = dynet.inputVector(mask)
                mask_expr = dynet.reshape(mask_expr, (1,), mb_size)
                loss = loss * mask_expr
            losses.append(loss)


        netloss = dynet.sum_batches(dynet.esum(losses))
        return netloss


    def sample(self, first=0, stop=-1, nchars=100):
        first = self.vocab[first].i
        stop = self.vocab[stop].i


        res = [first]
        dynet.renew_cg()
        state = self.rnn.initial_state()


        R = dynet.parameter(self.R)
        bias = dynet.parameter(self.bias)
        cw = first
        while True:
            x_t = self.lookup[cw]
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            scores = r_t.vec_value()
            if self.vocab.unk is not None:
                ydist = util.softmax(scores[:self.vocab.unk.i]+scores[self.vocab.unk.i+1:]) # remove UNK
                dist = ydist[:self.vocab.unk.i].tolist()+[0]+ydist[self.vocab.unk.i:].tolist()
            else:
                ydist = util.softmax(scores)
                dist = ydist
            rnd = random.random()
            for i,p in enumerate(dist):
                rnd -= p
                if rnd <= 0: break
            res.append(i)
            cw = i
            if cw == stop: break
            if nchars and len(res) > nchars: break
        return res


class BasicJointRNNLM(RNNLanguageModel):
    name = "joint"


    def __init__(self, model, vocab, args):
        self.m = model
        self.vocab = vocab
        self.args = args

        if args.s2s:
            print "loading s2s..."
            self.s2s = seq2seq.get_s2s(args.s2s_type).load(model, args.s2s)
			
        self.rnn = args.rnn(args.layers, self.s2s.args.hidden_dim*2 + args.input_dim, args.hidden_dim, model)


        self.lookup = model.add_lookup_parameters((vocab.size, args.input_dim))
        self.R = model.add_parameters((vocab.size, args.hidden_dim))
        self.bias = model.add_parameters((vocab.size,))


    def BuildLMGraph(self, sent, sent_args=None):
        dynet.renew_cg()
        init_state = self.rnn.initial_state()


        R = dynet.parameter(self.R)
        bias = dynet.parameter(self.bias)
        errs = [] # will hold expressions
        state = init_state


        for (cw,nw) in zip(sent,sent[1:]):
            cw = self.vocab[cw]
            nw = self.vocab[nw]


            spelling = [self.s2s.src_vocab[letter] for letter in cw.s.upper()]
            embedded_spelling = self.s2s.embed_seq(spelling)
            pron_vector = self.s2s.encode_seq(embedded_spelling)[-1]
            fpv = dynet.nobackprop(pron_vector)


            x_t = dynet.concatenate([self.lookup[cw.i], fpv])
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            err = dynet.pickneglogsoftmax(r_t, int(nw.i))
            errs.append(err)
        nerr = dynet.esum(errs)
        return nerr


    def BuildLMGraph_batch(self, sents, sent_args=None):
        dynet.renew_cg()
        init_state = self.rnn.initial_state()
        mb_size = len(sents)
        #MASK SENTENCES
        wids = [] # Dimension: maxSentLength * minibatch_size


        # List of lists to store whether an input is 
        # present(1)/absent(0) for an example at a time step
        masks = [] # Dimension: maxSentLength * minibatch_size


        #No of words processed in this batch
        tot_words = 0
        maxSentLength = max([len(sent) for sent in sents])


        for k in range(maxSentLength):
            wids.append([(self.vocab.s2t[sent[k]] if len(sent)>k else self.vocab.END_TOK) for sent in sents])
            mask = [(1 if len(sent)>k else 0) for sent in sents]
            masks.append(mask)
            tot_words += sum(mask)


        R = dynet.parameter(self.R)
        bias = dynet.parameter(self.bias)
        losses = [] # will hold losses
        state = init_state
        spellings = [] # list of lists containing spellings of the word

        for (mask, curr_words, next_words) in zip(masks, wids, wids[1:]):
            # print curr_words
            # print next_words
            maxWordLen = max([len(word.s) for word in curr_words])
            wordLengths = [len(word.s) for word in curr_words]

            for k in range(maxWordLen):
                spellings.append([(self.s2s.src_vocab[word.s[k].upper()].i if len(word.s)>k else self.s2s.src_vocab.END_TOK.i) for word in curr_words])

            spellings_rev = list(reversed(spellings))
            embedded_spellings = self.s2s.embed_batch_seq(spellings)
            embedded_spellings_rev = self.s2s.embed_batch_seq(spellings_rev)

            pron_vectors = self.s2s.encode_batch_seq(embedded_spellings, embedded_spellings_rev, wordLengths)[-1]
            
            fpv = dynet.nobackprop(pron_vectors)
            
            curr_words_idx = [word.i for word in curr_words]
            curr_words_lookup = dynet.lookup_batch(self.lookup, curr_words_idx)
            
            temp = dynet.concatenate([curr_words_lookup, fpv])
            x_t = temp
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            next_words_idx = [word.i for word in next_words]
            loss = dynet.pickneglogsoftmax_batch(r_t, next_words_idx)
            # loss is a list of losses
            # mask the loss if at least one sentence is shorter
            if 0 in mask:
                mask_expr = dynet.inputVector(mask)
                mask_expr = dynet.reshape(mask_expr, (1,), mb_size)
                loss = loss * mask_expr
            losses.append(loss)


        netloss = dynet.sum_batches(dynet.esum(losses))
        return netloss








    def sample(self, first=0, stop=-1, nchars=100):
        first = self.vocab[first].i
        stop = self.vocab[stop].i


        res = [first]
        dynet.renew_cg()
        state = self.rnn.initial_state()


        R = dynet.parameter(self.R)
        bias = dynet.parameter(self.bias)
        cw = first
        while True:
            spelling = [self.s2s.src_vocab[letter] for letter in self.vocab[cw].s.upper()]
            embedded_spelling = self.s2s.embed_seq(spelling)
            pron_vector = self.s2s.encode_seq(embedded_spelling)[-1]


            x_t = dynet.concatenate([self.lookup[cw], pron_vector])
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            scores = r_t.vec_value()
            if self.vocab.unk is not None:
                ydist = util.softmax(scores[:self.vocab.unk.i]+scores[self.vocab.unk.i+1:]) # remove UNK
                dist = ydist[:self.vocab.unk.i].tolist()+[0]+ydist[self.vocab.unk.i:].tolist()
            else:
                ydist = util.softmax(scores)
                dist = ydist
            rnd = random.random()
            for i,p in enumerate(dist):
                rnd -= p
                if rnd <= 0: break
            res.append(i)
            cw = i
            if cw == stop: break
            if nchars and len(res) > nchars: break
        return res


class PhonemeOnlyRNNLM(RNNLanguageModel):
    name = "prononly"


    def __init__(self, model, vocab, args):
        self.m = model
        self.vocab = vocab
        self.args = args

        if args.s2s:
            print "loading s2s..."
            self.s2s = seq2seq.get_s2s(args.s2s_type).load(model, args.s2s)
			
        self.rnn = args.rnn(args.layers, self.s2s.args.hidden_dim*2, args.hidden_dim, model)


        self.R = model.add_parameters((vocab.size, args.hidden_dim))
        self.bias = model.add_parameters((vocab.size,))


    def BuildLMGraph(self, sent, sent_args=None):
        dynet.renew_cg()
        init_state = self.rnn.initial_state()


        R = dynet.parameter(self.R)
        bias = dynet.parameter(self.bias)
        errs = [] # will hold expressions
        state = init_state


        for (cw,nw) in zip(sent,sent[1:]):
            cw = self.vocab[cw]
            nw = self.vocab[nw]


            spelling = [self.s2s.src_vocab[letter] for letter in cw.s.upper()]
            embedded_spelling = self.s2s.embed_seq(spelling)
            pron_vector = self.s2s.encode_seq(embedded_spelling)[-1]
            fpv = dynet.nobackprop(pron_vector)


            x_t = fpv
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            err = dynet.pickneglogsoftmax(r_t, int(nw.i))
            errs.append(err)
        nerr = dynet.esum(errs)
        return nerr


    def BuildLMGraph_batch(self, sents, sent_args=None):
        dynet.renew_cg()
        init_state = self.rnn.initial_state()
        mb_size = len(sents)
        #MASK SENTENCES
        wids = [] # Dimension: maxSentLength * minibatch_size


        # List of lists to store whether an input is 
        # present(1)/absent(0) for an example at a time step
        masks = [] # Dimension: maxSentLength * minibatch_size


        #No of words processed in this batch
        tot_words = 0
        maxSentLength = max([len(sent) for sent in sents])
        sentLengths =[len(sent) for sent in sents]


        for k in range(maxSentLength):
            wids.append([(self.vocab.s2t[sent[k]].i if len(sent)>k else self.vocab.END_TOK.i) for sent in sents])
            mask = [(1 if len(sent)>k else 0) for sent in sents]
            masks.append(mask)
            tot_words += sum(mask)


        R = dynet.parameter(self.R)
        bias = dynet.parameter(self.bias)
        losses = [] # will hold losses
        state = init_state


        for (mask, curr_words, next_words) in zip(masks,wids,wids[1:]):
            # print "Current words: ", curr_words
            # print "Next words: ", next_words
            x_t = dynet.lookup_batch(self.lookup, curr_words) 
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            loss = dynet.pickneglogsoftmax_batch(r_t, next_words)
            # loss is a list of losses
            # mask the loss if at least one sentence is shorter
            if 0 in mask:
                mask_expr = dynet.inputVector(mask)
                mask_expr = dynet.reshape(mask_expr, (1,), mb_size)
                loss = loss * mask_expr
            losses.append(loss)


        netloss = dynet.sum_batches(dynet.esum(losses))
        return netloss


    def sample(self, first=0, stop=-1, nchars=100):
        first = self.vocab[first].i
        stop = self.vocab[stop].i


        res = [first]
        dynet.renew_cg()
        state = self.rnn.initial_state()


        R = dynet.parameter(self.R)
        bias = dynet.parameter(self.bias)
        cw = first
        while True:
            spelling = [self.s2s.src_vocab[letter] for letter in self.vocab[cw].s.upper()]
            embedded_spelling = self.s2s.embed_seq(spelling)
            pron_vector = self.s2s.encode_seq(embedded_spelling)[-1]


            x_t = pron_vector
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            scores = r_t.vec_value()
            if self.vocab.unk is not None:
                ydist = util.softmax(scores[:self.vocab.unk.i]+scores[self.vocab.unk.i+1:]) # remove UNK
                dist = ydist[:self.vocab.unk.i].tolist()+[0]+ydist[self.vocab.unk.i:].tolist()
            else:
                ydist = util.softmax(scores)
                dist = ydist
            rnd = random.random()
            for i,p in enumerate(dist):
                rnd -= p
                if rnd <= 0: break
            res.append(i)
            cw = i
            if cw == stop: break
            if nchars and len(res) > nchars: break
        return res




