from __future__ import division
from itertools import izip
import dynet
import random, math, os, util
import tsne as tsne
import pylab as Plot
import numpy as np
import distance
import cPickle as pickle
import nltk.translate.bleu_score as BLEU

class Seq2SeqTemplate(object):
    name = "template"

def get_s2s(name):
    for c in util.itersubclasses(Seq2SeqTemplate):
        if c.name == name: return c
    raise Exception("No seq2seq model found with name: " + name)

class Seq2SeqBasic(Seq2SeqTemplate):
    """
    Bidirectional LSTM encoder and unidirectional decoder without attention
    """
    name = "basic"

    def __init__(self, model, src_vocab, tgt_vocab, args):

        self.m = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.args = args

        # Bidirectional Encoder LSTM
        print "Adding Forward encoder LSTM parameters"
        self.enc_fwd_lstm = dynet.LSTMBuilder(args.layers, args.input_dim, args.hidden_dim, model)
        print "Adding Backward encoder LSTM parameters"
        self.enc_bwd_lstm = dynet.LSTMBuilder(args.layers, args.input_dim, args.hidden_dim, model)

        #Decoder LSTM
        print "Adding decoder LSTM parameters"
        self.dec_lstm = dynet.LSTMBuilder(args.layers, args.hidden_dim * 2, args.hidden_dim, model)

        #Decoder weight and bias
        print "Adding Decoder weight"
        self.decoder_w = model.add_parameters( (tgt_vocab.size, args.hidden_dim))
        print "Adding Decoder bias"
        self.decoder_b = model.add_parameters( (tgt_vocab.size,))

        #Lookup parameters
        print "Adding lookup parameters"

        self.src_lookup = model.add_lookup_parameters( (src_vocab.size, args.input_dim))
        self.tgt_lookup = model.add_lookup_parameters( (tgt_vocab.size, args.hidden_dim * 2)) #Not entirely sure

        # # Initialize embeddings
        # if args.init_embeddings_src:
        #     src_or_tgt = "src"
        #     self.src_lookup = embeddings.EmbeddingInitializer.initialize_embeddings(self, src_or_tgt, args, src_vocab, self.src_lookup)
        #
        # if args.init_embeddings_tgt:
        #     src_or_tgt = "tgt"
        #     self.tgt_lookup = embeddings.EmbeddingInitializer.initialize_embeddings(self, src_or_tgt, args, tgt_vocab, self.tgt_lookup)
        #
        # # Initialize parameters for bilingual loss
        # if args.loss_function == "bilingual":
        #     self.cooccurrence_src, self.cooccurrence_tgt = util.get_cooccurrence_matrix(train_data_src, train_data_tgt)
        #     self.word_alignments_fwd, self.word_alignments_bwd = util.get_alignment_matrix(train_data_src, train_data_tgt, src_vocab, tgt_vocab)
        #     self.glove_w_src = model.add_lookup_parameters(( self.src_vocab.size * 2, args.input_dim ))
        #     self.glove_b_src = model.add_parameters(( self.src_vocab.size * 2, ))
        #     self.glove_w_tgt = model.add_lookup_parameters(( self.tgt_vocab.size * 2, args.input_dim ))
        #     self.glove_b_tgt = model.add_parameters(( self.tgt_vocab.size * 2, ))


    def save(self, path):
        if not os.path.exists(path): os.makedirs(path)
        self.src_vocab.save(path+"/vocab.src")
        self.tgt_vocab.save(path+"/vocab.tgt")
        self.m.save(path+"/params")
        with open(path+"/args", "w") as f: pickle.dump(self.args, f)

    @classmethod
    def load(cls, model, path):
        if not os.path.exists(path): raise Exception("Model "+path+" does not exist")
        src_vocab = util.Vocab.load(path+"/vocab.src")
        tgt_vocab = util.Vocab.load(path+"/vocab.tgt")
        with open(path+"/args", "r") as f: args = pickle.load(f)
        s2s = cls(model, src_vocab, tgt_vocab, args)
        s2s.m.load(path+"/params")
        return s2s

    def embed_seq(self, seq):

        wembs = [self.src_lookup[self.src_vocab[tok].i] for tok in seq]
        return wembs

    def embed_batch_seq(self, wids):

        wembs_batch = [dynet.lookup_batch(self.src_lookup, wid) for wid in wids]
        return wembs_batch

    def encode_seq(self, src_seq):

        src_seq_rev = list(reversed(src_seq))
        fwd_vectors = self.enc_fwd_lstm.initial_state().transduce(src_seq)
        bwd_vectors = self.enc_bwd_lstm.initial_state().transduce(src_seq_rev)
        bwd_vectors = list(reversed(bwd_vectors))
        vectors = [dynet.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
        return vectors

    def encode_batch_seq(self, src_seq, src_seq_rev):
        ### POSSIBLE BUG IN THIS FUNCTION => MASK NOT USED : Need to fix
        # (padded tokens should not contribute to the last hidden state)

        # transduce returns list of output expressions for each time step
        fwd_vectors = self.enc_fwd_lstm.initial_state().transduce(src_seq)
        bwd_vectors = list(reversed(self.enc_bwd_lstm.initial_state().transduce(src_seq_rev)))
        return dynet.concatenate([fwd_vectors[-1], bwd_vectors[-1]])

    def decode(self, encoding, input, output, alpha_loss):
        src_toks = [self.src_vocab[tok] for tok in input]
        tgt_toks = [self.tgt_vocab[tok] for tok in output]

        w = dynet.parameter(self.decoder_w)
        b = dynet.parameter(self.decoder_b)
        s = self.dec_lstm.initial_state().add_input(encoding)
        loss = []

        if self.args.loss_function == "bilingual":
            bilingual_loss = embeddings.LossFunctions.bilingual_loss(True)

        sent = []
        for tok in tgt_toks:
            out_vector = dynet.affine_transform([b, w, s.output()])
            probs = dynet.softmax(out_vector)
            cross_ent_loss = -dynet.log(dynet.pick(probs, tok.i))
            if self.args.loss_function == "bilingual":
                selection = np.argmax(probs.value())
                sent.append(self.tgt_vocab[selection])
            loss.append(cross_ent_loss + alpha_loss * bilingual_loss)
            embed_vector = self.tgt_lookup[tok.i]
            s = s.add_input(embed_vector)

        loss = dynet.esum(loss)
        if self.args.loss_function == "bilingual":
            bilingual_loss = embeddings.LossFunctions.bilingual_loss(False, src_toks, sent)
            loss = loss + alpha_loss * bilingual_loss
        return loss


    def decode_batch(self, encoding, output_batch):

        """

        :param encoding: last hidden state from encoder
        :param output_batch: list of output sentences in format [word1, word2..]
        :return: loss
        """
        w = dynet.parameter(self.decoder_w)
        b = dynet.parameter(self.decoder_b)
        s = self.dec_lstm.initial_state().add_input(encoding)
        losses = []

        maxSentLength = max([len(sent) for sent in output_batch])
        wids = []
        masks = []
        for j in range(maxSentLength):
            wids.append([(self.tgt_vocab[sent[j]].i if len(sent)>j else self.tgt_vocab.END_TOK.i) for sent in output_batch])
            mask = [(1 if len(sent)>j else 0) for sent in output_batch]
            masks.append(mask)

        for wid, mask in zip(wids, masks):

            # calculate the softmax and loss
            score = dynet.affine_transform([b, w, s.output()])
            loss = dynet.pickneglogsoftmax_batch(score, wid)

            # mask the loss if at least one sentence is shorter than maxSentLength
            if 0 in mask:
                mask_expr = dynet.inputVector(mask)
                mask_expr = dynet.reshape(mask_expr, (1,), len(output_batch))
                loss = loss * mask_expr

            losses.append(loss)

            # update the state of the RNN
            embed_vector = dynet.lookup_batch(self.tgt_lookup, wid)
            s = s.add_input(embed_vector)

        return dynet.sum_batches(dynet.esum(losses))


    def generate(self, src, sampled=False):
        embedding = self.embed_seq(src)
        encoding = self.encode_seq(embedding)[-1]

        w = dynet.parameter(self.decoder_w)
        b = dynet.parameter(self.decoder_b)

        s = self.dec_lstm.initial_state().add_input(encoding)

        out = []
        for _ in range(5*len(src)):
            out_vector = dynet.affine_transform([b, w, s.output()])
            probs = dynet.softmax(out_vector)
            selection = np.argmax(probs.value())
            out.append(self.tgt_vocab[selection])
            if out[-1].s == self.tgt_vocab.END_TOK: break
            embed_vector = self.tgt_lookup[selection]
            s = s.add_input(embed_vector)
        return out

    def beam_search_generate(self, src_seq, beam_n=5):
        dynet.renew_cg()

        embedded = self.embed_seq(src_seq)
        input_vectors = self.encode_seq(embedded)

        w = dynet.parameter(self.decoder_w)
        b = dynet.parameter(self.decoder_b)

        s = self.dec_lstm.initial_state()
        s = s.add_input(input_vectors[-1])
        beams = [{"state":  s,
                  "out":    [],
                  "err":    0}]
        completed_beams = []
        while len(completed_beams) < beam_n:
            potential_beams = []
            for beam in beams:
                if len(beam["out"]) > 0:
                    embed_vector = self.tgt_lookup[beam["out"][-1].i]
                    s = beam["state"].add_input(embed_vector)

                out_vector = dynet.affine_transform([b, w, s.output()])
                probs = dynet.softmax(out_vector)
                probs = probs.vec_value()

                for potential_next_i in range(len(probs)):
                    potential_beams.append({"state":    s,
                                            "out":      beam["out"]+[self.tgt_vocab[potential_next_i]],
                                            "err":      beam["err"]-math.log(probs[potential_next_i])})

            potential_beams.sort(key=lambda x:x["err"])
            beams = potential_beams[:beam_n-len(completed_beams)]
            completed_beams = completed_beams+[beam for beam in beams if beam["out"][-1] == self.tgt_vocab.END_TOK
                                                                      or len(beam["out"]) > 5*len(src_seq)]
            beams = [beam for beam in beams if beam["out"][-1] != self.tgt_vocab.END_TOK
                                            and len(beam["out"]) <= 5*len(src_seq)]
        completed_beams.sort(key=lambda x:x["err"])
        return [beam["out"] for beam in completed_beams]


    def get_loss(self, input, output, alpha_loss):
        dynet.renew_cg()
        embedded = self.embed_seq(input)
        encoded = self.encode_seq(embedded)[-1]
        return self.decode(encoded, input, output, alpha_loss)


    def get_batch_loss(self, input_batch, output_batch):

        dynet.renew_cg()

        # Dimension: maxSentLength * minibatch_size
        wids = []
        wids_reversed = []

        # List of lists to store whether an input is
        # present(1)/absent(0) for an example at a time step
        masks = [] # Dimension: maxSentLength * minibatch_size

        #No of words processed in this batch
        tot_words = 0
        maxSentLength = max([len(sent) for sent in input_batch])
        sent_lengths =[len(sent) for sent in input_batch]

        for j in range(maxSentLength):
            wids.append([(self.src_vocab[sent[j]].i if len(sent)>j else self.src_vocab.END_TOK.i) for sent in input_batch])
            wids_reversed.append([(self.src_vocab[sent[len(sent)-j-1]].i if len(sent)>j else self.src_vocab.END_TOK.i) for sent in input_batch])
            mask = [(1 if len(sent)>j else 0) for sent in input_batch]
            masks.append(mask)
            tot_words += sum(mask)

        embedded_batch = self.embed_batch_seq(wids)
        embedded_batch_reverse = self.embed_batch_seq(wids_reversed)
        encoded_batch = self.encode_batch_seq(embedded_batch, embedded_batch_reverse)

        # pass last hidden state of encoder to decoder
        return self.decode_batch(encoded_batch, output_batch)

    def get_bleu(self, input, output, beam_n=5):
        guess = self.generate(input, sampled=False)
        input_str = [tok.s for tok in guess]
        output_str = [tok.s for tok in output]
        ans = BLEU.sentence_bleu(input_str, output_str)
        return ans

    def get_em(self, input, output, beam_n=5):
        guess = self.generate(input, sampled=False)
        input_str = [tok.s for tok in guess]
        output_str = [tok.s for tok in output]
        ans = 1 if input_str == output_str else 0
        return ans

    def tsne_embeddings(self):

        perplexity = 20.0
        wembs_tgt = [self.tgt_lookup[i].npvalue() for i in range(self.tgt_vocab.size)]

        Y = tsne.tsne(np.array(wembs_tgt), 2, self.args.input_dim, perplexity)
        indices = [i for i in range(self.tgt_vocab.size)]
        words = [self.tgt_vocab.i2t[i].s for i in range(self.tgt_vocab.size)]
        Plot.scatter(Y[:,0], Y[:,1], 20, indices)

        for word, x, y in zip(words, Y[:, 0], Y[:, 1]):
          Plot.annotate(word, xy = (x, y))

        Plot.show()



class Seq2SeqBiRNNAttn(Seq2SeqBasic):
    """
    Bidirectional LSTM encoder and unidirectional decoder with attention
    """
    name="attention"

    def __init__(self, model, src_vocab, tgt_vocab, args):
        self.m = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.args = args
        # Bidirectional Encoder LSTM
        print "Adding Forward encoder LSTM parameters"
        self.enc_fwd_lstm = dynet.LSTMBuilder(args.layers, args.input_dim, args.hidden_dim, model)
        print "Adding Backward encoder LSTM parameters"
        self.enc_bwd_lstm = dynet.LSTMBuilder(args.layers, args.input_dim, args.hidden_dim, model)

        #Decoder LSTM
        print "Adding decoder LSTM parameters"
        self.dec_lstm = dynet.LSTMBuilder(args.layers, args.hidden_dim*2 + args.hidden_dim*2, args.hidden_dim, model)

        #Decoder weight and bias
        print "Adding Decoder weight"
        self.decoder_w = model.add_parameters( (tgt_vocab.size, args.hidden_dim))
        print "Adding Decoder bias"
        self.decoder_b = model.add_parameters( (tgt_vocab.size,))

        print "Adding lookup parameters"
        #Lookup parameters
        self.src_lookup = model.add_lookup_parameters( (src_vocab.size, args.input_dim))
        self.tgt_lookup = model.add_lookup_parameters( (tgt_vocab.size, args.hidden_dim * 2))

        #Attention parameters
        print "Adding Attention Parameters"
        self.attention_w1 = model.add_parameters( (args.attention_dim, args.hidden_dim * 2))
        self.attention_w2 = model.add_parameters( (args.attention_dim, args.hidden_dim*args.layers * 2))
        self.attention_v = model.add_parameters( (1, args.attention_dim))

    def attend(self, input_vectors, state):
        w1 = dynet.parameter(self.attention_w1)
        w2 = dynet.parameter(self.attention_w2)
        v = dynet.parameter(self.attention_v)
        attention_weights = []
        w2dt = w2 * dynet.concatenate(list(state.s()))

        for input_vector in input_vectors:
            attention_weight = v * dynet.tanh(w1 * input_vector + w2dt)
            attention_weights.append(attention_weight)
        attention_weights = dynet.softmax(dynet.concatenate(attention_weights))
        output_vectors = dynet.esum([vector*attention_weight for vector, attention_weight in zip(input_vectors, attention_weights)])
        return output_vectors

    def decode(self, input_vectors, output):
        tgt_toks = [self.tgt_vocab[tok] for tok in output]

        w = dynet.parameter(self.decoder_w)
        b = dynet.parameter(self.decoder_b)

        s = self.dec_lstm.initial_state()
        s = s.add_input(dynet.concatenate([
                                            input_vectors[-1],
                                            dynet.vecInput(self.args.hidden_dim*2)
                                          ]))
        loss = []
        for tok in tgt_toks:
            out_vector = dynet.affine_transform([b, w, s.output()])
            probs = dynet.softmax(out_vector)
            loss.append(-dynet.log(dynet.pick(probs, tok.i)))
            embed_vector = self.tgt_lookup[tok.i]
            attn_vector = self.attend(input_vectors, s)
            inp = dynet.concatenate([embed_vector, attn_vector])
            s = s.add_input(inp)

        loss = dynet.esum(loss)
        return loss


    def decode_batch(self, encoding, output_batch):

        w = dynet.parameter(self.decoder_w)
        b = dynet.parameter(self.decoder_b)
        s = self.dec_lstm.initial_state()
        losses = []

        maxSentLength = max([len(sent) for sent in output_batch])
        wids = []
        masks = []
        for j in range(maxSentLength):
            wids.append([(self.tgt_vocab[sent[j]].i if len(sent)>j else self.tgt_vocab.END_TOK.i) for sent in output_batch])
            mask = [(1 if len(sent)>j else 0) for sent in output_batch]
            masks.append(mask)

        s = s.add_input(dynet.concatenate([
                                            encoding[-1],
                                            dynet.vecInput(self.args.hidden_dim*2)
                                          ]))
        for wid, mask in zip(wids, masks):

            # calculate the softmax and loss
            score = dynet.affine_transform([b, w, s.output()])
            loss = dynet.pickneglogsoftmax_batch(score, wid)

            # mask the loss if at least one sentence is shorter
            if 0 in mask:
                mask_expr = dynet.inputVector(mask)
                mask_expr = dynet.reshape(mask_expr, (1,), len(output_batch))
                loss = loss * mask_expr

            losses.append(loss)

            # update the state of the RNN
            embed_vector = dynet.lookup_batch(self.tgt_lookup, wid)
            attn_vector = self.attend(encoding, s)
            inp = dynet.concatenate([embed_vector, attn_vector])
            s = s.add_input(inp)

        return dynet.sum_batches(dynet.esum(losses))

    def generate(self, src_seq, sampled=False):
        def sample(probs):
            rnd = random.random()
            for i, p in enumerate(probs):
                rnd -= p
                if rnd <= 0: break
            return i

        dynet.renew_cg()

        embedded = self.embed_seq(src_seq)
        input_vectors = self.encode_seq(embedded)


        w = dynet.parameter(self.decoder_w)
        b = dynet.parameter(self.decoder_b)

        s = self.dec_lstm.initial_state()
        s = s.add_input(dynet.concatenate([
                                            input_vectors[-1],
                                            dynet.vecInput(self.args.hidden_dim*2)
                                          ]))
        out = []
        for i in range(1+len(src_seq)*5):
            out_vector = dynet.affine_transform([b, w, s.output()])
            probs = dynet.softmax(out_vector)
            probs = probs.vec_value()
            next_symbol = sample(probs) if sampled else max(enumerate(probs), key=lambda x:x[1])[0]
            out.append(self.tgt_vocab[next_symbol])
            if self.tgt_vocab[next_symbol] == self.tgt_vocab.END_TOK:
                break
            embed_vector = self.tgt_lookup[out[-1].i]
            attn_vector = self.attend(input_vectors, s)
            inp = dynet.concatenate([embed_vector, attn_vector])
            s = s.add_input(inp)
        return out, input_vectors[-1].npvalue()

    def beam_search_generate(self, src_seq, beam_n=5):
        dynet.renew_cg()

        embedded = self.embed_seq(src_seq)
        input_vectors = self.encode_seq(embedded)

        w = dynet.parameter(self.decoder_w)
        b = dynet.parameter(self.decoder_b)

        s = self.dec_lstm.initial_state()
        s = s.add_input(dynet.concatenate([
                                            input_vectors[-1],
                                            dynet.vecInput(self.args.hidden_dim*2)
                                          ]))
        beams = [{"state":  s,
                  "out":    [],
                  "err":    0}]
        completed_beams = []
        while len(completed_beams) < beam_n:
            potential_beams = []
            for beam in beams:
                if len(beam["out"]) > 0:
                    attn_vector = self.attend(input_vectors, beam["state"])
                    embed_vector = self.tgt_lookup[beam["out"][-1].i]
                    inp = dynet.concatenate([embed_vector, attn_vector])
                    s = beam["state"].add_input(inp)

                out_vector = dynet.affine_transform([b, w, s.output()])
                probs = dynet.softmax(out_vector)
                probs = probs.vec_value()

                for potential_next_i in range(len(probs)):
                    potential_beams.append({"state":    s,
                                            "out":      beam["out"]+[self.tgt_vocab[potential_next_i]],
                                            "err":      beam["err"]-math.log(probs[potential_next_i])})

            potential_beams.sort(key=lambda x:x["err"])
            beams = potential_beams[:beam_n-len(completed_beams)]
            completed_beams = completed_beams+[beam for beam in beams if beam["out"][-1] == self.tgt_vocab.END_TOK
                                                                      or len(beam["out"]) > 5*len(src_seq)]
            beams = [beam for beam in beams if beam["out"][-1] != self.tgt_vocab.END_TOK
                                            and len(beam["out"]) <= 5*len(src_seq)]
        completed_beams.sort(key=lambda x:x["err"])
        return [beam["out"] for beam in completed_beams]


    def encode_batch_seq(self, src_seq, src_seq_rev):

        fwd_vectors = self.enc_fwd_lstm.initial_state().transduce(src_seq)
        bwd_vectors = self.enc_bwd_lstm.initial_state().transduce(src_seq_rev)
        return [dynet.concatenate(list(p)) for p in zip(fwd_vectors, list(reversed(bwd_vectors)))]

    def get_batch_loss(self, input_batch, output_batch):

        dynet.renew_cg()

        # Dimension: maxSentLength * minibatch_size
        wids = []
        wids_reversed = []

        # List of lists to store whether an input is
        # present(1)/absent(0) for an example at a time step
        masks = [] # Dimension: maxSentLength * minibatch_size

        #No of words processed in this batch
        tot_words = 0
        maxSentLength = max([len(sent) for sent in input_batch])

        for j in range(maxSentLength):
            wids.append([(self.src_vocab[sent[j]].i if len(sent)>j else self.src_vocab.END_TOK.i) for sent in input_batch])
            wids_reversed.append([(self.src_vocab[sent[len(sent)-j-1]].i if len(sent)>j else self.src_vocab.END_TOK.i) for sent in input_batch])
            mask = [(1 if len(sent)>j else 0) for sent in input_batch]
            masks.append(mask)
            tot_words += sum(mask)

        embedded_batch = self.embed_batch_seq(wids)
        embedded_batch_reverse = self.embed_batch_seq(wids_reversed)
        encoded_batch = self.encode_batch_seq(embedded_batch, embedded_batch_reverse)

        # pass all hidden states of encoder to decoder (for attention)
        return self.decode_batch(encoded_batch, output_batch)

    def get_loss(self, input, output):
        dynet.renew_cg()
        embedded = self.embed_seq(input)
        encoded = self.encode_seq(embedded)
        return self.decode(encoded, output)

    def get_bleu(self, input, output, beam_n=5):
        guesses = self.beam_search_generate(input, beam_n)
        input_strs = [[tok.s for tok in guess] for guess in guesses]
        output_strs = [tok.s for tok in output]
        ans = max([BLEU.sentence_bleu(input_str, output_strs) for input_str in input_strs])
        return ans


    def tsne_embeddings(self, test_data):

        vecs = []
        words = []
        perplexity = 20.0

        for src, target in test_data:
            dynet.renew_cg()
            symbols, last_hidden_state = self.generate(src)
            symbols = [symbol.s for symbol in symbols if symbol!=self.tgt_vocab.END_TOK]
            words.append(src)
            vecs.append(last_hidden_state)

        Y = tsne.tsne(np.array(vecs), 2, self.args.input_dim, perplexity)
        indices = [i for i in range(len(words))]
        Plot.scatter(Y[:,0], Y[:,1], 20, indices)

        for word, x, y in zip(words, Y[:, 0], Y[:, 1]):
          Plot.annotate(word, xy = (x, y))

        Plot.show()
