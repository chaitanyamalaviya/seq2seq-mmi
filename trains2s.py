import dynet
import argparse
import util
import time
import random
import sys
import seq2seq

# input args

parser = argparse.ArgumentParser()

## need to have this dummy argument for dynet
parser.add_argument("--dynet-mem")
parser.add_argument("--dynet-gpu")

## locations of data
parser.add_argument("--train")
parser.add_argument("--valid")
parser.add_argument("--test")
parser.add_argument("--plot_embeddings")

## alternatively, load one dataset and split it
parser.add_argument("--percent_valid", default=1000, type=float)

## vocab parameters
parser.add_argument('--rebuild_vocab', action='store_true')
parser.add_argument('--unk_thresh', default=1, type=int)

## rnn parameters
parser.add_argument("--layers", default=1, type=int)
parser.add_argument("--input_dim", default=10, type=int)
parser.add_argument("--hidden_dim", default=50, type=int)
parser.add_argument("--attention_dim", default=50, type=int)

## experiment parameters
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--learning_rate", default=1.0, type=float)
parser.add_argument("--log_train_every_n", default=100, type=int)
parser.add_argument("--log_valid_every_n", default=5000, type=int)
parser.add_argument("--output")

## choose what model to use
parser.add_argument("--model", default="basic")
parser.add_argument("--reader_mode")
parser.add_argument("--pronouncer")
parser.add_argument("--pronouncer_type", default="basic")
parser.add_argument("--load")
parser.add_argument("--save")
parser.add_argument("--eval", type=bool)

## model-specific parameters
parser.add_argument("--beam_size", default=3, type=int)

args = parser.parse_args()
print "ARGS:", args

BEGIN_TOKEN = '<s>'
END_TOKEN = '<e>'

# define model

model = dynet.Model()
sgd = dynet.SimpleSGDTrainer(model)

S2SModel = seq2seq.get_s2s(args.model)
if args.load:
    print "Loading model..."
    s2s = S2SModel.load(model, args.load)
    src_vocab = s2s.src_vocab
    tgt_vocab = s2s.tgt_vocab
else:
    print "fresh model. getting vocab...",
    src_reader = util.get_reader(args.reader_mode)(args.train, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN)
    src_vocab = util.Vocab.load_from_corpus(src_reader, remake=args.rebuild_vocab, src_or_tgt="src")
    src_vocab.START_TOK = src_vocab[BEGIN_TOKEN]
    src_vocab.END_TOK = src_vocab[END_TOKEN]
    src_vocab.add_unk(args.unk_thresh)
    tgt_reader = util.get_reader(args.reader_mode)(args.train, mode=args.reader_mode, end=END_TOKEN)
    tgt_vocab = util.Vocab.load_from_corpus(tgt_reader, remake=args.rebuild_vocab, src_or_tgt="tgt")
    tgt_vocab.END_TOK = tgt_vocab[END_TOKEN]
    tgt_vocab.add_unk(args.unk_thresh)
    print "making model..."
    s2s = S2SModel(model, src_vocab, tgt_vocab, args)
    print "...done."


# evaluate existing model
if args.eval:
    print "Evaluating model..."
    test_data = list(util.get_reader(args.reader_mode)(args.test, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN))
    # if args.test:
    #     s2s.evaluate(test_data)  
    #     sys.exit("...done.")
    # else:
    #     raise Exception("Test file path argument missing")
    
    if args.test and args.plot_embeddings:
        s2s.tsne_embeddings(test_data)
        sys.exit("...done.")





# load corpus

print "loading corpus..."
train_data = list(util.get_reader(args.reader_mode)(args.train, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN))
if args.valid:
    valid_data = list(util.get_reader(args.reader_mode)(args.valid, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN))
else:
    if args.percent_valid > 1: cutoff = args.percent_valid
    else: cutoff = int(len(train_data)*(args.percent_valid))
    valid_data = train_data[-cutoff:]
    train_data = train_data[:-cutoff]
    print "Train set of size", len(train_data), "/ Validation set of size", len(valid_data)
print "done."

if args.eval:
    print "Evaluating model..."
    train_data = list(util.get_reader(args.reader_mode)(args.train, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN))
    if args.test:
        s2s.evaluate(train_data)
        sys.exit("...done.")
    else:
        raise Exception("Test file path argument missing")


if args.output:
    outfile = open(args.output, 'w')
    outfile.write("")
    outfile.close()



# run training loop

char_count = sent_count = cum_loss = 0.0
_start = time.time()
try:
    for ITER in range(args.epochs):
        s2s.epoch = ITER
        random.shuffle(train_data)

        for i,(src,tgt) in enumerate(train_data):
            src = [src_vocab[s] for s in src]
            tgt = [tgt_vocab[s] for s in tgt]
            sample_num = 1+i+(len(train_data)*ITER)
            # print sample_num, src_vocab.pp(src, ' '), tgt_vocab.pp(tgt, ' ')

            if sample_num % args.log_train_every_n == 0:
                print ITER, sample_num, " ",
                sgd.status()
                print "L:", cum_loss / char_count if char_count != 0 else None,
                print "T:", (time.time() - _start),
                _start = time.time()
                # sample = lm.beam_search_generate(src, beam_n=args.beam_size)
                sample = s2s.generate(src, sampled=False)
                if sample: print src_vocab.pp(src, ' '), tgt_vocab.pp(tgt, ' '), tgt_vocab.pp(sample, ' '),
                char_count = sent_count = cum_loss = cum_bleu = 0.0
                print
            # end of test logging

            if sample_num % args.log_valid_every_n == 0:
                v_char_count = v_sent_count = v_cum_loss = v_cum_bleu = v_cum_em = v_cum_perp = 0.0
                v_start = time.time()
                for v_src, v_tgt in valid_data:
                    v_src = [src_vocab[tok] for tok in v_src]
                    v_tgt = [tgt_vocab[tok] for tok in v_tgt]
                    v_loss = s2s.get_loss(v_src, v_tgt)
                    v_cum_loss += v_loss.scalar_value()
                    v_cum_perp += s2s.get_perplexity(v_src, v_tgt)
                    v_cum_em += s2s.get_em(v_src, v_tgt)
                    # v_cum_bleu += s2s.get_bleu(v_src, v_tgt, args.beam_size)
                    v_char_count += len(v_tgt)-1
                    v_sent_count += 1
                print "[Validation "+str(sample_num) + "]\t" + \
                      "Loss: "+str(v_cum_loss / v_char_count) + "\t" + \
                      "Perp: "+str(v_cum_perp / v_sent_count) + "\t" + \
                      "BLEU: "+str(v_cum_bleu / v_sent_count) + "\t" + \
                      "EM: "  +str(v_cum_em   / v_sent_count) + "\t" + \
                      "Time: "+str(time.time() - v_start),
                if args.output:
                    print "(logging to", args.output + ")"
                    with open(args.output, "a") as outfile:
                        outfile.write(str(ITER) + "\t" + \
                                      str(sample_num) + "\t" + \
                                      str(v_cum_loss / v_char_count) + "\t" + \
                                      str(v_cum_perp / v_sent_count) + "\t" + \
                                      str(v_cum_em   / v_sent_count) + "\t" + \
                                      str(v_cum_bleu / v_sent_count) + "\n")
                print "\n"
                if args.save:
                    print "saving checkpoint..."
                    s2s.save(args.save+".checkpoint")
            # end of validation logging

            loss = s2s.get_loss(src, tgt)
            cum_loss += loss.value()
            char_count += len(tgt)-1
            sent_count += 1

            loss.backward()
            sgd.update(args.learning_rate)

            ### end of one-sentence train loop
        sgd.update_epoch(args.learning_rate)
        ### end of iteration
    ### end of training loop
except KeyboardInterrupt:
    if args.save:
        print "saving..."
        s2s.save(args.save)
        sys.exit()

if args.save:
    print "saving..."
    s2s.save(args.save)
